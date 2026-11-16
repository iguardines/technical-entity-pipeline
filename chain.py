"""Cadena LCEL asíncrona, validación de salida y reintentos acotados."""
import logging
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError
import openai
import anthropic

from errors import InvalidModelOutput, ModelRefusal, PipelineError, TruncatedOutput
from mock_model import DemoChatModel
from prompts import PROMPT
from schemas import TechnicalEntities

logger = logging.getLogger(__name__)
RETRYABLE = (InvalidModelOutput, OutputParserException, ValidationError,
             openai.LengthFinishReasonError, openai.APIConnectionError, openai.RateLimitError,
             anthropic.APIConnectionError, anthropic.RateLimitError)


def validate_output(result: dict) -> TechnicalEntities:
    """Inspecciona metadata antes de aceptar/transformar el objeto final.

    include_raw conserva errores del parser interno en parsing_error.
    Nunca aceptamos una respuesta truncada aunque el parser haya recuperado JSON.
    """
    raw = result.get('raw')
    metadata = getattr(raw, 'response_metadata', {})
    if metadata.get('finish_reason') in ('length', 'max_tokens') or metadata.get('stop_reason') == 'max_tokens':
        raise TruncatedOutput('Respuesta truncada por el límite de tokens.')
    additional = getattr(raw, 'additional_kwargs', {})
    content = getattr(raw, 'content', [])
    if metadata.get('finish_reason') == 'content_filter' or additional.get('refusal') or (
        isinstance(content, list) and any(isinstance(part, dict) and part.get('type') == 'refusal' for part in content)
    ):
        raise ModelRefusal('El proveedor rechazó la solicitud.')
    if result.get('parsing_error') is not None:
        raise InvalidModelOutput('JSON inválido o incompatible con el esquema.') from result['parsing_error']
    if result.get('parsed') is None:
        raise InvalidModelOutput('El proveedor no devolvió un objeto estructurado.')
    try:
        return TechnicalEntities.model_validate(result['parsed'])
    except ValidationError as error:
        raise InvalidModelOutput('La salida no satisface el contrato Pydantic.') from error


def build_chain(model):
    # El parser Pydantic está incluido en with_structured_output.
    return PROMPT | model.with_structured_output(TechnicalEntities, include_raw=True) | RunnableLambda(validate_output)


def create_model():
    load_dotenv(Path(__file__).with_name('.env'))
    mode = os.getenv('LLM_USE_MOCK', 'true').lower()
    if mode not in ('true', 'false'):
        raise PipelineError('LLM_USE_MOCK debe ser true o false.')
    if mode == 'true':
        return DemoChatModel()
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    if provider not in ('openai', 'anthropic'):
        raise PipelineError('LLM_PROVIDER debe ser openai o anthropic.')
    key_name = 'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY'
    if not os.getenv(key_name, '').strip():
        raise PipelineError('Falta configurar ' + key_name + '.')
    model_name = os.getenv('LLM_MODEL', '').strip()
    if not model_name:
        raise PipelineError('Falta configurar LLM_MODEL.')
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=None, max_tokens=4096, max_retries=0, timeout=45)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model_name, temperature=None, max_tokens=4096, max_retries=0, timeout=45)


async def process_text(text: str, *, model=None, max_attempts: int = 3, retry_wait: bool = True) -> TechnicalEntities:
    """Devuelve un objeto validado; 3 intentos totales por defecto (2 reintentos).

    model permite inyectar un modelo de prueba. El contador es local a cada llamada,
    de modo que ejecuciones concurrentes no mezclan intentos.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('El texto de entrada no puede estar vacío.')
    if len(text) > 20000:
        raise ValueError('El texto de entrada no puede superar 20000 caracteres.')
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 2 <= max_attempts <= 5:
        raise ValueError('max_attempts debe estar entre 2 y 5.')
    request_id, attempt = uuid4().hex[:8], 0
    try:
        chain = build_chain(model if model is not None else create_model())

        async def invoke_attempt(values):
            nonlocal attempt
            attempt += 1
            logger.info('solicitud=%s intento=%s/%s', request_id, attempt, max_attempts)
            try:
                result = await chain.ainvoke(values)
            except Exception as error:
                retry = isinstance(error, RETRYABLE) and attempt < max_attempts
                logger.warning('solicitud=%s intento=%s error=%s reintento=%s',
                               request_id, attempt, type(error).__name__, retry)
                raise
            logger.info('solicitud=%s validacion=ok tecnologias=%s', request_id, len(result.tecnologias))
            return result

        resilient_chain = RunnableLambda(invoke_attempt).with_retry(
            retry_if_exception_type=RETRYABLE,
            stop_after_attempt=max_attempts,
            wait_exponential_jitter=retry_wait,
        )
        return await resilient_chain.ainvoke({'text': text.strip()})
    except PipelineError:
        raise
    except Exception as error:
        # No incluir mensajes del SDK: pueden contener el texto de entrada.
        raise PipelineError('No se pudo completar el procesamiento (' + type(error).__name__ + ').') from error

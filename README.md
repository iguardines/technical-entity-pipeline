# Pipeline de extracción de entidades técnicas

Pre-entrega 2 — Pipeline de procesamiento validado, Python 3.12.

Recibe un texto de arquitectura o un log y devuelve un objeto Pydantic con
`tecnologias`, `nivel_de_criticidad` y `resumen_tecnico`. Usa LangChain, un
`ChatPromptTemplate`, LCEL, `.with_structured_output()` y `.with_retry()`.

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

En Windows, activar con `.venv\Scripts\activate`.

## Probar sin claves ni costo

El modo predeterminado es local (`LLM_USE_MOCK=true`):

```bash
python main.py
python main.py --demo-retry
python main.py --text "Algo anda raro, pero no tengo más detalles."
python -m unittest discover -s tests -v
```

`--demo-retry` fuerza el modelo local aunque `.env` esté configurado para llamadas
reales. Genera JSON mal formado en el primer intento y una respuesta válida en el
segundo. Los logs permiten observar la recuperación.

El mock es una demostración determinista que reconoce una lista pequeña de
tecnologías. **No es un LLM ni reproduce la comprensión de un proveedor real.**
Sin embargo, utiliza la cadena LCEL y el parser Pydantic reales de LangChain;
las pruebas no reemplazan la validación ni la implementación de los reintentos.

## Ejemplo

Entrada:

> La API usa FastAPI, Redis como caché y PostgreSQL como persistencia. El pool de conexiones está agotado y provoca una caída total del servicio.

Ejemplo ilustrativo de salida válida (la redacción de un LLM puede variar):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; caída del servicio por agotamiento del pool de conexiones."
}
```

El mock identifica las mismas tecnologías y criticidad del ejemplo, pero utiliza
un resumen prefijado con `Demostración local:` para no confundirlo con salida real.

## Contrato y textos ambiguos

- Todos los campos son obligatorios y se rechazan campos adicionales.
- `tecnologias` contiene strings no vacíos; elimina espacios y duplicados.
- Se permite `tecnologias: []` cuando el texto no identifica ninguna. Rechazar esa
  lista obligaría a inventar entidades o a fallar por una entrada legítima.
- `nivel_de_criticidad` es un enum: `baja`, `media`, `alta`.
- El resumen debe tener entre 10 y 1000 caracteres después de quitar espacios.
- El texto de entrada no puede estar vacío ni superar 20000 caracteres.
- El prompt define criterios de criticidad. Sin impacto demostrado, solicita
  criticidad baja y un resumen que reconozca la falta de contexto.

Pydantic valida la estructura, no garantiza que el contenido sea verdadero ni
que la criticidad inferida sea correcta. Eso requiere evaluación con ejemplos reales.

## Cadena y resiliencia

La composición central en `chain.py` es:

```python
PROMPT | model.with_structured_output(TechnicalEntities, include_raw=True) | RunnableLambda(validate_output)
```

`with_structured_output` incluye el parser Pydantic. Se conserva la respuesta
original para inspeccionar `finish_reason` (OpenAI) y `stop_reason` (Anthropic).
Si indican `length` o `max_tokens`, se rechaza el resultado, incluso si el parser
interno consiguió recuperar un objeto. Esta comprobación precede a la aceptación
y transformación final. También se maneja `LengthFinishReasonError` del SDK.

El bloque completo se ejecuta dentro de un `RunnableLambda` con `.with_retry()`:
**tres intentos totales, hasta dos reintentos**, con espera exponencial y jitter.
Cada intento vuelve a llamar al modelo; no se vuelve a parsear la misma respuesta.
Se reintentan JSON mal formado, campos faltantes, validación fallida, truncamiento,
errores de conexión y límites de tasa. No se reintentan rechazos explícitos del
modelo ni errores de autenticación. Los SDK tienen sus propios reintentos
deshabilitados para evitar multiplicar llamadas inadvertidamente.

Si se agotan los intentos, se devuelve un error controlado (`PipelineError` o
subclases). Los reintentos no garantizan la recuperación: si el límite de tokens
sigue siendo insuficiente, hay que reducir el texto o ajustar el presupuesto.
El mini-script sale con código 1 ante errores y no imprime un traceback.

Los logs incluyen identificador de solicitud, número de intento, tipo de error,
si se reintentará y resultado de validación. No registran claves, texto de entrada
ni respuestas completas. El JSON se escribe en stdout y los logs en stderr.

## Uso asíncrono desde Python

```python
import asyncio
from chain import process_text

async def ejemplo():
    resultado = await process_text("FastAPI usa Redis y PostgreSQL sin incidentes.")
    print(resultado.model_dump_json(indent=2))

asyncio.run(ejemplo())
```

La función utiliza `.ainvoke()`. Para pruebas se puede inyectar `model` y desactivar
las esperas con `retry_wait=False`. El contador de intentos es local a cada llamada,
lo que permite solicitudes concurrentes sin mezclar sus contadores.

## Activar un proveedor real

Se conserva la convención de configuración del TP1; no se importa ni depende
de su carpeta. Los adaptadores de LangChain (`ChatOpenAI` / `ChatAnthropic`)
reemplazan a los clientes directos para poder componer LCEL.

Configurar `.env`:

```env
LLM_USE_MOCK=false
LLM_PROVIDER=openai
LLM_MODEL=identificador-del-modelo-disponible-en-tu-cuenta
OPENAI_API_KEY=tu-clave-local
```

Para Anthropic, usar `LLM_PROVIDER=anthropic`, el modelo correspondiente y
`ANTHROPIC_API_KEY`. Elegir un modelo compatible con salida estructurada según
su proveedor. La temperatura se omite para evitar restricciones de modelos que
no permiten ajustarla. El presupuesto de salida es 4096 tokens y el timeout 45 s.

```bash
python main.py
```

Una ejecución real puede generar hasta tres solicitudes facturables por los
reintentos. Los valores del entorno tienen prioridad sobre `.env`. No se cargan
claves del TP1 automáticamente. `.env` y `.venv/` están excluidos de Git; incluir
únicamente `.env.example` con valores vacíos.

## Pruebas y alcance de la validación

La suite usa `unittest`, sin herramientas de testing adicionales ni claves.
Cubre contrato, prompts, selección/configuración de proveedores, éxito asíncrono,
JSON inválido e incompleto, límite de reintentos, truncamiento, rechazo del modelo,
texto ambiguo, concurrencia, cancelación y logs sin datos de entrada.

Se construyen los adaptadores reales de ambos proveedores con claves ficticias,
sin enviar solicitudes. **Las llamadas reales de este TP todavía no están
verificadas.** Las pruebas de integración del TP1 no sustituyen esa comprobación.

## Archivos

```text
schemas.py           Contrato Pydantic y enum
prompts.py           ChatPromptTemplate e instrucciones de formato
chain.py             LCEL, proveedores, validación, reintentos y process_text
errors.py            Errores controlados
mock_model.py        Modelo local para demo y pruebas
main.py              Mini-script asíncrono
requirements.txt     Dependencias directas verificadas
requirements.lock   Versiones completas del entorno de validación
.env.example         Variables de entorno sin secretos
.gitignore           Exclusiones para Git
 tests/test_pipeline.py
```

Para reproducir todas las versiones del entorno de validación, instalar
`requirements.lock` en lugar de `requirements.txt`.

## Referencias

- [Salida estructurada en LangChain](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output)
- [Reintentos de Runnable](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/with_retry)

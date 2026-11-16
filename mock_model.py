"""Modelo local de demostración; usa el parser real de LangChain.

No es un extractor inteligente: reconoce una lista pequeña de tecnologías.
"""
import asyncio
import json
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


class DemoChatModel(BaseChatModel):
    fail_first: bool = False
    calls: int = 0
    scripted: list[AIMessage] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'local-demo'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _reply(self, messages: list[BaseMessage]) -> ChatResult:
        self.calls += 1
        if self.scripted:
            message = self.scripted[min(self.calls - 1, len(self.scripted) - 1)]
        elif self.fail_first and self.calls == 1:
            message = tool_message('{JSON inválido')
        else:
            text = str(messages[-1].content)
            technologies = [name for name in ('FastAPI', 'Redis', 'PostgreSQL', 'Python', 'Docker') if name.casefold() in text.casefold()]
            critical = any(word in text.casefold() for word in ('caída', 'indisponible', 'pérdida de datos'))
            payload = {
                'tecnologias': technologies,
                'nivel_de_criticidad': 'alta' if critical else 'baja',
                'resumen_tecnico': ('Demostración local: ' + text[:350]) if technologies else 'Contexto insuficiente: no se identifican tecnologías explícitas.',
            }
            message = tool_message(json.dumps(payload, ensure_ascii=False))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._reply(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(0)
        return self._reply(messages)


def tool_message(arguments: str, finish_reason: str = 'stop') -> AIMessage:
    return AIMessage(content='', response_metadata={'finish_reason': finish_reason},
        additional_kwargs={'tool_calls': [{'id': 'demo-call', 'type': 'function',
            'function': {'name': 'TechnicalEntities', 'arguments': arguments}}]})

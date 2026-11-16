"""Contrato de salida del extractor técnico."""
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Criticidad(StrEnum):
    BAJA = 'baja'
    MEDIA = 'media'
    ALTA = 'alta'


class TechnicalEntities(BaseModel):
    model_config = ConfigDict(extra='forbid')

    tecnologias: list[str] = Field(description='Tecnologías explícitas; lista vacía si no se identifican.')
    nivel_de_criticidad: Criticidad
    resumen_tecnico: str = Field(description='Resumen breve en español basado únicamente en el texto.')

    @field_validator('tecnologias')
    @classmethod
    def clean_technologies(cls, values: list[str]) -> list[str]:
        result, seen = [], set()
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError('Una tecnología no puede estar vacía.')
            if value.casefold() not in seen:
                result.append(value)
                seen.add(value.casefold())
        return result

    @field_validator('resumen_tecnico')
    @classmethod
    def validate_summary(cls, value: str) -> str:
        value = value.strip()
        if not 10 <= len(value) <= 1000:
            raise ValueError('El resumen debe tener entre 10 y 1000 caracteres.')
        return value

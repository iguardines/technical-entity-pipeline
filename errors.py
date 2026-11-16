class PipelineError(Exception):
    """Error controlado expuesto al consumidor del pipeline."""


class InvalidModelOutput(PipelineError):
    """JSON inválido, campos faltantes o restricciones incumplidas."""


class TruncatedOutput(InvalidModelOutput):
    """El proveedor agotó los tokens antes de completar la respuesta."""


class ModelRefusal(PipelineError):
    """El modelo rechazó responder; no se reintenta automáticamente."""

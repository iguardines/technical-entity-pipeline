"""Mini-script asíncrono. JSON por stdout; logs por stderr."""
import argparse
import asyncio
import logging
import sys

from chain import process_text
from errors import PipelineError
from mock_model import DemoChatModel

EXAMPLE = ('La API usa FastAPI, Redis como caché y PostgreSQL como persistencia. '
           'El pool de conexiones está agotado y provoca una caída total del servicio.')


async def main() -> int:
    parser = argparse.ArgumentParser(description='Extracción validada de entidades técnicas')
    parser.add_argument('--text', default=EXAMPLE)
    parser.add_argument('--demo-retry', action='store_true', help='Simular un JSON inválido y recuperación, sin API')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
    # Evita logs HTTP de terceros; el pipeline registra intentos y validación.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    model = DemoChatModel(fail_first=True) if args.demo_retry else None
    try:
        result = await process_text(args.text, model=model)
    except (PipelineError, ValueError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))

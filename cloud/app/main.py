"""FastAPI Cloud service exposing the versioned PipeSD protocol."""

import argparse
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cloud.core.tasks import TaskManager
from cloud.models import LlamaCppTargetBackend, MockTargetBackend
from shared.protocol import ProtocolError, error_payload
from shared.serialization import CONTENT_TYPE, pack_message, unpack_message
from shared.version import PROTOCOL_VERSION


def create_app(backend=None, task_ttl_s=600):
    app = FastAPI(title="PipeSD Cloud", version=PROTOCOL_VERSION)
    manager = TaskManager(backend or MockTargetBackend(), ttl_s=task_ttl_s)
    app.state.task_manager = manager

    async def decode(request: Request):
        return unpack_message(await request.body())

    def reply(payload, status_code=200):
        return Response(pack_message(payload), status_code=status_code, media_type=CONTENT_TYPE)

    @app.exception_handler(ProtocolError)
    async def protocol_error(_request, exc):
        return reply(error_payload(str(exc)), 409)

    @app.exception_handler(Exception)
    async def internal_error(_request, exc):
        # Keep the wire format stable even for unexpected backend failures.
        return reply(error_payload(f"Cloud verification failed: {type(exc).__name__}: {exc}"), 500)

    @app.get("/health")
    async def health():
        manager.cleanup_expired()
        return {"status": "ok", "protocol_version": PROTOCOL_VERSION, "active_tasks": len(manager.tasks)}

    @app.post("/init")
    async def init(request: Request):
        return reply(manager.init_task(await decode(request)))

    @app.post("/propose")
    async def propose(request: Request):
        manager.cleanup_expired()
        return reply(manager.propose(await decode(request)))

    @app.post("/exit")
    async def exit_task(request: Request):
        payload = await decode(request)
        return reply(manager.exit_task(payload["task_id"]))

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="PipeSD Cloud target verifier")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--target-model-path", default=os.environ.get("PIPE_SD_TARGET_MODEL", ""))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.mock and not args.target_model_path:
        parser.error("--target-model-path is required unless --mock is used")
    backend = MockTargetBackend() if args.mock else LlamaCppTargetBackend(args.target_model_path)
    import uvicorn
    uvicorn.run(create_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

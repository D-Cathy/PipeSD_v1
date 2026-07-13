"""FastAPI Cloud service exposing the versioned PipeSD protocol."""

import argparse
import importlib
import json
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cloud.core.tasks import TaskManager
from cloud.models import (
    LlamaCppTargetBackend, MockTargetBackend, MockVideoTargetBackend,
    Qwen3VLTargetBackend,
)
from cloud.core.video_tasks import VideoTaskManager
from shared.protocol import ProtocolError, error_payload
from shared.serialization import CONTENT_TYPE, pack_message, unpack_message
from shared.version import PROTOCOL_VERSION


def create_app(backend=None, task_ttl_s=600, video_backend=None):
    app = FastAPI(title="PipeSD Cloud", version=PROTOCOL_VERSION)
    manager = TaskManager(backend or MockTargetBackend(), ttl_s=task_ttl_s)
    video_manager = VideoTaskManager(video_backend or MockVideoTargetBackend(), ttl_s=task_ttl_s)
    app.state.task_manager = manager
    app.state.video_task_manager = video_manager

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
        video_manager.cleanup_expired()
        return {
            "status": "ok", "protocol_version": PROTOCOL_VERSION,
            "active_tasks": len(manager.tasks), "active_video_tasks": len(video_manager.tasks),
        }

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

    @app.post("/video/init")
    async def video_init(request: Request):
        return reply(video_manager.init_task(await decode(request)))

    @app.post("/video/propose")
    async def video_propose(request: Request):
        video_manager.cleanup_expired()
        return reply(video_manager.propose(await decode(request)))

    @app.post("/video/exit")
    async def video_exit(request: Request):
        payload = await decode(request)
        return reply(video_manager.exit_task(payload["task_id"]))

    return app


app = create_app()


def load_factory(spec):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("Backend factory must use module:callable syntax.")
    return getattr(importlib.import_module(module_name), attribute)


def main():
    parser = argparse.ArgumentParser(description="PipeSD Cloud target verifier")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--target-model-path", default=os.environ.get("PIPE_SD_TARGET_MODEL", ""))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video-backend-factory", help="Optional module:callable returning a VideoTargetBackend")
    parser.add_argument("--video-backend-kwargs", default="{}")
    parser.add_argument("--video-target-model-path", help="Local Qwen3-VL Cloud model directory")
    parser.add_argument("--video-device", default="cuda:0")
    parser.add_argument("--video-allow-cpu", action="store_true")
    args = parser.parse_args()
    if not args.mock and not args.target_model_path:
        parser.error("--target-model-path is required unless --mock is used")
    backend = MockTargetBackend() if args.mock else LlamaCppTargetBackend(args.target_model_path)
    if args.video_backend_factory and args.video_target_model_path:
        parser.error("Use either --video-backend-factory or --video-target-model-path, not both")
    if args.video_backend_factory:
        video_backend = load_factory(args.video_backend_factory)(**json.loads(args.video_backend_kwargs))
    elif args.video_target_model_path:
        video_backend = Qwen3VLTargetBackend(
            args.video_target_model_path, device=args.video_device,
            allow_cpu=args.video_allow_cpu,
        )
    else:
        video_backend = MockVideoTargetBackend()
    import uvicorn
    uvicorn.run(create_app(backend, video_backend=video_backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

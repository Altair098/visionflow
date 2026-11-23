from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from visionflow.config import AppConfig
from visionflow.pipeline import PipelineManager


def create_app(cfg: AppConfig) -> FastAPI:
    manager = PipelineManager(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        yield
        manager.stop()

    app = FastAPI(title="VisionFlow", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "cameras": len(manager.pipelines)}

    @app.get("/cameras")
    def cameras():
        return manager.states()

    @app.get("/cameras/{camera_id}")
    def camera(camera_id: str):
        states = manager.states()
        if camera_id not in states:
            raise HTTPException(404, f"Unknown camera: {camera_id}")
        return states[camera_id]

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app

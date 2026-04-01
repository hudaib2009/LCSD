from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import ALLOW_ORIGINS, HEATMAP_DIR
from backend.app.services.inference import (
    InferenceError,
    cxr_embeddings_upload,
    generate_plan_payload,
    health_payload,
    infer_case,
    predict_ct_payload,
    predict_pathology_upload,
    predict_xray_upload,
    startup_models,
)

logger = logging.getLogger("csd.api")


def create_app() -> FastAPI:
    app = FastAPI(title="CSD Inference Service")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        startup_models()

    app.mount("/static", StaticFiles(directory=HEATMAP_DIR), name="static")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_payload()

    @app.post("/predict/xray")
    async def predict_xray(file: UploadFile = File(...)) -> JSONResponse:
        return execute(lambda: predict_xray_upload(file))

    @app.post("/predict/ct")
    async def predict_ct(request: Request) -> JSONResponse:
        payload = await request.json()
        return execute(lambda: predict_ct_payload(payload))

    @app.post("/predict/pathology")
    async def predict_pathology(file: UploadFile = File(...)) -> JSONResponse:
        return execute(lambda: predict_pathology_upload(file))

    @app.post("/plan")
    def generate_plan(payload: dict[str, Any]) -> JSONResponse:
        return execute(lambda: generate_plan_payload(payload))

    @app.post("/api/embeddings/cxr")
    async def cxr_embeddings_endpoint(file: UploadFile = File(...)) -> JSONResponse:
        status_code, payload = cxr_embeddings_upload(file)
        return JSONResponse(status_code=status_code, content=payload)

    @app.post("/infer")
    def infer(payload: dict[str, Any]) -> JSONResponse:
        return execute(lambda: infer_case(payload))

    return app


def execute(fn) -> JSONResponse:
    try:
        return JSONResponse(status_code=200, content=fn())
    except InferenceError as exc:
        logger.warning("%s: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "detail": exc.detail,
            },
        )
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Inference failed.",
                "detail": str(exc),
            },
        )


app = create_app()

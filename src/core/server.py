import logging

from fastapi import FastAPI
from fastapi.middleware import Middleware

from src import api
from src.core.fastapi.middleware.logging import LoggingMiddleware

logger = logging.getLogger(__name__)


def init_routers(_app: FastAPI) -> None:
    _app.include_router(api.router)


def make_middleware() -> list[Middleware]:
    middleware = [
        Middleware(LoggingMiddleware),
    ]
    return middleware


def create_app() -> FastAPI:
    _app = FastAPI(
        title="Bot-Detector-API",
        description="Bot-Detector-API",
        middleware=make_middleware(),
    )
    init_routers(_app=_app)
    return _app


app = create_app()


@app.get("/")
async def root():
    return {"message": "Hello World"}

"""Main FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat
from app.utils.config import config
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    logger.info(
        "Starting MCS-Agent",
        extra={
            "host": config.server_host,
            "port": config.server_port,
        },
    )
    yield
    logger.info(
        "Shutting down MCS-Agent",
        extra={
            "host": config.server_host,
            "port": config.server_port,
        },
    )


app = FastAPI(
    title="MCS-Agent",
    description="A self-improving AI agent using Monte Carlo Tree Search",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "MCS-Agent"}

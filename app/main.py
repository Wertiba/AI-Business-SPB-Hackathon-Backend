from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.audio import router as audio_router

app = FastAPI(
    title="Ping",
    description="Ping",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redocs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(audio_router, prefix="/api/v1")


@app.get("/ping")
async def ping():
    return {"message": "pong"}

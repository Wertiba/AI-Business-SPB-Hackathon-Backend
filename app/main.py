from app.api import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

origins = ["localhost"]
app = FastAPI(
    title="AI Business SPB",
    version="1.0",
    description="API for case \"Engine problems anomaly detection\"",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)
Instrumentator().instrument(app).expose(app)


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"message": "pong"}

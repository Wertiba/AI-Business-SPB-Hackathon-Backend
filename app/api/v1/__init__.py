from fastapi import APIRouter

from .audio import router as audio_router

router = APIRouter(prefix="/v1")

router.include_router(audio_router)

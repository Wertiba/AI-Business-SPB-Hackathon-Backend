from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.services.audio_classifier import audio_classifier

router = APIRouter(prefix="/audio", tags=["audio"])


class AudioClassificationResponse(BaseModel):
    result: bool
    message: str


@router.post("/classify", response_model=AudioClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only .mp3 files are accepted")

    audio_bytes = await file.read()
    classification = await audio_classifier.classify(audio_bytes)
    return AudioClassificationResponse(result=classification.result, message=classification.message)
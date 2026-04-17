from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.audio_model import AudioClassificationResponse

from app.services.audio_classifier import audio_classifier

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/classify", response_model=AudioClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")

    audio_bytes = await file.read()
    classification = await audio_classifier.classify(audio_bytes)
    return AudioClassificationResponse(
        result=classification.result, message=classification.message
    )

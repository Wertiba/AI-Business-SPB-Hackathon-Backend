from app.schemas.audio import AudioClassificationResponse, BatchAudioClassificationResponse
from app.services.audio_service import AudioService
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/audio", tags=["audio"])
audio_service = AudioService()


@router.post("/classify", response_model=AudioClassificationResponse)
async def classify_audio(file: UploadFile = File()) -> AudioClassificationResponse:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")
    return await audio_service.classify_single(file)


@router.post("/classify-batch", response_model=BatchAudioClassificationResponse)
async def classify_audio_batch(
    file: UploadFile = File(description="ZIP archive containing WAV files"),
) -> BatchAudioClassificationResponse:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")
    return await audio_service.classify_batch_zip(file)

from enum import StrEnum
from typing import Annotated

from app.schemas.audio import BatchResponse, ClassificationResponse, SegmentType
from app.services.audio_service import AudioService
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/audio", tags=["audio"])
audio_service = AudioService()


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify a single WAV file",
)
async def classify_audio(
    files: list[UploadFile] = File(description="WAV file, 48kHz 24-bit stereo"),
    vehicle_id: Annotated[str | None, Form(description="Vehicle identifier")] = None,
    segment_type: Annotated[SegmentType, Form(description="idle / high_hold / background")] = SegmentType.IDLE,
    duration_sec: Annotated[float | None, Form(description="Segment duration in seconds")] = None,
) -> ClassificationResponse:
    if len(files) > 1:
        raise HTTPException(status_code=400, detail="Only one file is allowed.")

    file = files[0]
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")

    return await audio_service.classify_single(file, vehicle_id, segment_type, duration_sec)


@router.post(
    "/classify-batch",
    response_model=BatchResponse,
    summary="Classify a batch of WAV files from ZIP",
)
async def classify_audio_batch(
    files: list[UploadFile] = File(description="ZIP archive containing WAV files"),
    vehicle_id: Annotated[str | None, Form(description="Vehicle identifier")] = None,
    segment_type: Annotated[SegmentType, Form(description="idle / high_hold / background")] = SegmentType.IDLE,
) -> BatchResponse:
    if len(files) > 1:
        raise HTTPException(status_code=400, detail="Only one ZIP file is allowed per request.")

    file = files[0]
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    return await audio_service.classify_batch_zip(file, vehicle_id, segment_type)

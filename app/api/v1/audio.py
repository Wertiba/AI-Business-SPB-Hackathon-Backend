from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from typing import Annotated
from app.schemas.audio import BatchResponse, ClassificationResponse, SegmentType
from app.services.audio_service import AudioService

router = APIRouter(prefix="/audio", tags=["audio"])
audio_service = AudioService()


@router.post("/classify", response_model=ClassificationResponse)
async def classify_audio(
    request: Request,
    file: UploadFile = File(description="WAV file, 48kHz 24-bit stereo"),
    vehicle_id: Annotated[str | None, Form(description="Vehicle identifier")] = None,
    segment_type: Annotated[SegmentType, Form(description="idle / high_hold / background")] = SegmentType.IDLE,
    duration_sec: Annotated[float | None, Form(description="Segment duration in seconds")] = None,
) -> ClassificationResponse:
    form = await request.form()
    if len(form.getlist("file")) > 1:
        raise HTTPException(status_code=400, detail="Only one file is allowed.")

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")

    return await audio_service.classify_single(file, vehicle_id, segment_type, duration_sec)


@router.post("/classify-batch", response_model=BatchResponse)
async def classify_audio_batch(
    request: Request,
    file: UploadFile = File(description="ZIP archive containing WAV files"),
    vehicle_id: Annotated[str | None, Form(description="Vehicle identifier")] = None,
    segment_type: Annotated[SegmentType, Form(description="idle / high_hold / background")] = SegmentType.IDLE,
) -> BatchResponse:
    form = await request.form()
    if len(form.getlist("file")) > 1:
        raise HTTPException(status_code=400, detail="Only one ZIP file is allowed per request.")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    return await audio_service.classify_batch_zip(file, vehicle_id, segment_type)

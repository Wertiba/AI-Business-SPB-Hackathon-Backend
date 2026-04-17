import io
import zipfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.audio_model import (
    AudioClassificationResponse,
    AudioClassificationItem,
    BatchAudioClassificationResponse,
)

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


@router.post("/classify-batch", response_model=BatchAudioClassificationResponse)
async def classify_audio_batch(
    file: UploadFile = File(..., description="ZIP archive containing WAV files"),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    zip_bytes = await file.read()

    try:
        zip_buffer = io.BytesIO(zip_bytes)
        zip_file = zipfile.ZipFile(zip_buffer, "r")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    wav_files = [
        name for name in zip_file.namelist()
        if name.lower().endswith(".wav") and not name.startswith("__MACOSX")
    ]

    if len(wav_files) > 10000:
        raise HTTPException(
            status_code=400, detail="Maximum 10000 WAV files allowed in ZIP"
        )

    if not wav_files:
        raise HTTPException(status_code=400, detail="No WAV files found in ZIP archive")

    items: list[AudioClassificationItem] = []
    successful = 0
    failed = 0

    for filename in wav_files:
        try:
            audio_bytes = zip_file.read(filename)
            classification = await audio_classifier.classify(audio_bytes)
            items.append(
                AudioClassificationItem(
                    filename=filename,
                    result=classification.result,
                    message=classification.message,
                )
            )
            successful += 1
        except Exception as e:
            items.append(
                AudioClassificationItem(
                    filename=filename,
                    result=False,
                    message=f"Error processing file: {str(e)}",
                )
            )
            failed += 1

    zip_file.close()

    return BatchAudioClassificationResponse(
        items=items,
        total=len(wav_files),
        successful=successful,
        failed=failed,
    )

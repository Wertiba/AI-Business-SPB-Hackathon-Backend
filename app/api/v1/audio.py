import asyncio
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.audio_model import (
    AudioClassificationResponse,
    AudioClassificationItem,
    BatchAudioClassificationResponse,
)

from app.services.audio_classifier import audio_classifier

router = APIRouter(prefix="/audio", tags=["audio"])

MAX_ZIP_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
MAX_FILES = 10000
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for reading


@router.post("/classify", response_model=AudioClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")

    audio_bytes = await file.read()
    classification = await audio_classifier.classify(audio_bytes)
    return AudioClassificationResponse(
        result=classification.result, message=classification.message
    )


def _extract_wav_from_zip(zip_path: str, filename: str) -> bytes:
    """Sync function to extract a single file from ZIP (runs in thread pool)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(filename)


def _get_wav_files_from_zip(zip_path: str) -> list[str]:
    """Get list of WAV files from ZIP (runs in thread pool)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [
            name for name in zf.namelist()
            if name.lower().endswith(".wav") and not name.startswith("__MACOSX")
        ]


@router.post("/classify-batch", response_model=BatchAudioClassificationResponse)
async def classify_audio_batch(
    file: UploadFile = File(..., description="ZIP archive containing WAV files"),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        total_size = 0
        while chunk := await file.read(CHUNK_SIZE):
            total_size += len(chunk)
            if total_size > MAX_ZIP_SIZE:
                raise HTTPException(
                    status_code=400, detail=f"ZIP file too large. Maximum size is {MAX_ZIP_SIZE // (1024**3)}GB"
                )
            tmp.write(chunk)
        tmp_path = tmp.name

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    try:
        wav_files = await loop.run_in_executor(executor, _get_wav_files_from_zip, tmp_path)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    if len(wav_files) > MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_FILES} WAV files allowed in ZIP"
        )

    if not wav_files:
        raise HTTPException(status_code=400, detail="No WAV files found in ZIP archive")

    items: list[AudioClassificationItem] = []
    successful = 0
    failed = 0

    for filename in wav_files:
        try:
            audio_bytes = await loop.run_in_executor(
                executor, _extract_wav_from_zip, tmp_path, filename
            )
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

    os.unlink(tmp_path)
    executor.shutdown(wait=False)

    return BatchAudioClassificationResponse(
        items=items,
        total=len(wav_files),
        successful=successful,
        failed=failed,
    )

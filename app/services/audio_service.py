import asyncio
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.schemas.audio import (
    BatchResponse,
    ClassificationItem,
    ClassificationResponse,
)
from app.services.classifier import audio_classifier
from app.core.metrics import anomaly_detected_total, anomaly_score_hist, files_processed_total
from fastapi import HTTPException, UploadFile

MAX_ZIP_SIZE = 5 * 1024 * 1024 * 1024
MAX_FILES = 10000
CHUNK_SIZE = 8 * 1024 * 1024


class AudioService:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    async def classify_single(file: UploadFile) -> ClassificationResponse:
        audio_bytes = await file.read()
        classification = await audio_classifier.classify(audio_bytes)

        anomaly_score_hist.observe(classification.anomaly_score)
        if classification.result:
            anomaly_detected_total.inc()
        files_processed_total.labels(status="success").inc()

        return ClassificationResponse(
            result=classification.result,
            message=classification.message,
            anomaly_score=classification.anomaly_score,
        )

    async def classify_batch_zip(self, file: UploadFile) -> BatchResponse:
        tmp_path = await self._save_zip(file)
        try:
            return await self._process_zip(tmp_path)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    async def _save_zip(file: UploadFile) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            total_size = 0
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_ZIP_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"ZIP file too large. Maximum size is {MAX_ZIP_SIZE // (1024**3)}GB",
                    )
                tmp.write(chunk)
            return tmp.name

    async def _process_zip(self, tmp_path: str) -> BatchResponse:
        loop = asyncio.get_running_loop()

        try:
            wav_files = await loop.run_in_executor(
                self._executor, self._get_wav_files, tmp_path
            )
        except zipfile.BadZipFile as e:
            raise HTTPException(status_code=400, detail="Invalid ZIP file") from e

        if not wav_files:
            raise HTTPException(status_code=400, detail="No WAV files found in ZIP archive")
        if len(wav_files) > MAX_FILES:
            raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} WAV files allowed")

        items: list[ClassificationItem] = []
        successful = 0
        failed = 0

        batch_size = audio_classifier.model.batch_size

        for i in range(0, len(wav_files), batch_size):
            batch_filenames = wav_files[i: i + batch_size]

            batch_bytes: list[tuple[str, bytes]] = []
            for filename in batch_filenames:
                try:
                    data = await loop.run_in_executor(
                        self._executor, self._read_from_zip, tmp_path, filename
                    )
                    batch_bytes.append((filename, data))
                except Exception as e:
                    items.append(ClassificationItem(
                        filename=filename,
                        result=False,
                        message=f"Error reading file: {e}",
                        anomaly_score=0.0,
                    ))
                    files_processed_total.labels(status="error").inc()
                    failed += 1

            if not batch_bytes:
                continue

            tmp_paths: list[Path] = []
            filename_map: dict[Path, str] = {}

            for filename, data in batch_bytes:
                try:
                    with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                    ) as tmp:
                        tmp.write(data)
                        p = Path(tmp.name)
                    tmp_paths.append(p)
                    filename_map[p] = filename
                except Exception as e:
                    items.append(ClassificationItem(
                        filename=filename,
                        result=False,
                        message=f"Error writing tmp file: {e}",
                        anomaly_score=0.0,
                    ))
                    files_processed_total.labels(status="error").inc()
                    failed += 1

            try:
                classifications = await audio_classifier.classify_batch(tmp_paths)
                for path, cls in zip(tmp_paths, classifications, strict=True):
                    items.append(ClassificationItem(
                        filename=filename_map[path],
                        result=cls.result,
                        message=cls.message,
                        anomaly_score=cls.anomaly_score,
                    ))
                    anomaly_score_hist.observe(cls.anomaly_score)
                    if cls.result:
                        anomaly_detected_total.inc()
                    files_processed_total.labels(status="success").inc()
                    successful += 1
            except Exception as e:
                for path in tmp_paths:
                    items.append(ClassificationItem(
                        filename=filename_map[path],
                        result=False,
                        message=f"Error processing: {e}",
                        anomaly_score=0.0,
                    ))
                    files_processed_total.labels(status="error").inc()
                    failed += 1
            finally:
                for path in tmp_paths:
                    path.unlink(missing_ok=True)

        return BatchResponse(
            items=items,
            total=len(wav_files),
            successful=successful,
            failed=failed,
        )

    @staticmethod
    def _get_wav_files(zip_path: str) -> list[str]:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return [
                name for name in zf.namelist()
                if name.lower().endswith(".wav") and not name.startswith("__MACOSX")
            ]

    @staticmethod
    def _read_from_zip(zip_path: str, filename: str) -> bytes:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.read(filename)

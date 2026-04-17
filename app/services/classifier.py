import tempfile
from pathlib import Path

from app.schemas.audio import ClassificationResponse
from solution import PredictionModel

THRESHOLD = 0.5


class AudioClassifierService:
    def __init__(self):
        self.model = PredictionModel()
        self.threshold = THRESHOLD

    async def classify(self, audio_bytes: bytes) -> ClassificationResponse:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            scores = self.model.predict([tmp_path])
            anomaly_score = scores[0] if scores else 0.0
        finally:
            tmp_path.unlink(missing_ok=True)

        is_anomaly = anomaly_score > self.threshold
        return ClassificationResponse(
            result=is_anomaly,
            message="Anomaly detected" if is_anomaly else "Normal",
            anomaly_score=anomaly_score,
        )

    async def classify_batch(self, file_paths: list[Path]) -> list[ClassificationResponse]:
        scores = self.model.predict(file_paths)
        results = []
        for score in scores:
            is_anomaly = score > self.threshold
            results.append(
                ClassificationResponse(
                    result=is_anomaly,
                    message="Anomaly detected" if is_anomaly else "Normal",
                    anomaly_score=score,
                )
            )
        return results


audio_classifier = AudioClassifierService()

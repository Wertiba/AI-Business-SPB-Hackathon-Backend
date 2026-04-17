import tempfile
from dataclasses import dataclass
from pathlib import Path

from solution import PredictionModel


@dataclass
class ClassificationResult:
    result: bool
    message: str
    anomaly_score: float


class AudioClassifierService:
    def __init__(self):
        self.model = PredictionModel()
        self.threshold = 0.5

    async def classify(self, audio_bytes: bytes) -> ClassificationResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            scores = self.model.predict([tmp_path])
            anomaly_score = scores[0] if scores else 0.0
        finally:
            tmp_path.unlink(missing_ok=True)

        is_anomaly = anomaly_score > self.threshold
        return ClassificationResult(
            result=is_anomaly,
            message="Anomaly detected" if is_anomaly else "Normal",
            anomaly_score=anomaly_score,
        )

    async def classify_batch(self, file_paths: list[Path]) -> list[ClassificationResult]:
        scores = self.model.predict(file_paths)
        results = []
        for score in scores:
            is_anomaly = score > self.threshold
            results.append(
                ClassificationResult(
                    result=is_anomaly,
                    message="Anomaly detected" if is_anomaly else "Normal",
                    anomaly_score=score,
                )
            )
        return results


audio_classifier = AudioClassifierService()
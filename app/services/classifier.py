import tempfile
from pathlib import Path

from app.schemas.audio import AnomalyLabel, ClassificationResponse
from solution import PredictionModel

THRESHOLD_SUSPICIOUS = 0.35
THRESHOLD_ANOMALY = 0.50
MODEL_VERSION = "1.0.0"


class AudioClassifierService:
    def __init__(self) -> None:
        self.model = PredictionModel()

    @staticmethod
    def score_to_label(score: float) -> AnomalyLabel:
        if score < THRESHOLD_SUSPICIOUS:
            return AnomalyLabel.NORMAL
        elif score < THRESHOLD_ANOMALY:
            return AnomalyLabel.SUSPICIOUS
        else:
            return AnomalyLabel.ANOMALY

    @staticmethod
    def normalize_score(raw: float, lo: float = 0.0, hi: float = 0.8) -> float:
        return float(min(max((raw - lo) / (hi - lo), 0.0), 1.0))

    async def classify(self, audio_bytes: bytes) -> ClassificationResponse:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            scores = self.model.predict([tmp_path])
            raw = scores[0] if scores else 0.0
        finally:
            tmp_path.unlink(missing_ok=True)

        score = self.normalize_score(raw)
        return ClassificationResponse(
            anomaly_score=score,
            label=self.score_to_label(score),
            rpm_estimate=None,
            model_version=MODEL_VERSION,
        )

    async def classify_batch(self, file_paths: list[Path]) -> list[ClassificationResponse]:
        scores = self.model.predict(file_paths)
        results = []
        for raw in scores:
            score = self.normalize_score(raw)
            results.append(ClassificationResponse(
                anomaly_score=score,
                label=self.score_to_label(score),
                rpm_estimate=None,
                model_version=MODEL_VERSION,
            ))
        return results


audio_classifier = AudioClassifierService()

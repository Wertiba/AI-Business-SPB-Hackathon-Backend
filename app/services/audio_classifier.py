from dataclasses import dataclass


@dataclass
class ClassificationResult:
    result: bool
    message: str


class AudioClassifierService:
    async def classify(self, audio_bytes: bytes) -> ClassificationResult:
        # TODO: replace with real model inference
        return self._fallback()

    def _fallback(self) -> ClassificationResult:
        return ClassificationResult(result=False, message="Fallback: model not available")


audio_classifier = AudioClassifierService()
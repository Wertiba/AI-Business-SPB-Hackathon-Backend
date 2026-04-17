from pydantic import BaseModel


class AudioClassificationResponse(BaseModel):
    result: bool
    message: str
    anomaly_score: float


class AudioClassificationItem(BaseModel):
    filename: str
    result: bool
    message: str
    anomaly_score: float


class BatchAudioClassificationResponse(BaseModel):
    items: list[AudioClassificationItem]
    total: int
    successful: int
    failed: int

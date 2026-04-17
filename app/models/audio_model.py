from pydantic import BaseModel


class AudioClassificationResponse(BaseModel):
    result: bool
    message: str


class AudioClassificationItem(BaseModel):
    filename: str
    result: bool
    message: str


class BatchAudioClassificationResponse(BaseModel):
    items: list[AudioClassificationItem]
    total: int
    successful: int
    failed: int

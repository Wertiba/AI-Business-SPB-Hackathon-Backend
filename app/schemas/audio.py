from pydantic import BaseModel


class ClassificationResponse(BaseModel):
    result: bool
    message: str
    anomaly_score: float


class ClassificationItem(BaseModel):
    filename: str
    result: bool
    message: str
    anomaly_score: float


class BatchResponse(BaseModel):
    items: list[ClassificationItem]
    total: int
    successful: int
    failed: int

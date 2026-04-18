from enum import StrEnum
from pydantic import BaseModel, Field


class SegmentType(StrEnum):
    IDLE = "idle"
    HIGH_HOLD = "high_hold"
    BACKGROUND = "background"


class AnomalyLabel(StrEnum):
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ANOMALY = "anomaly"


class ClassificationResponse(BaseModel):
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Anomaly score [0..1]")
    label: AnomalyLabel = Field(description="normal / suspicious / anomaly")
    rpm_estimate: float | None = Field(default=None, description="Estimated RPM if available")
    model_version: str = Field(default="1.0.0")


class ClassificationItem(BaseModel):
    filename: str
    anomaly_score: float = Field(ge=0.0, le=1.0)
    label: AnomalyLabel
    rpm_estimate: float | None = None
    model_version: str = "1.0.0"
    error: str | None = None


class BatchResponse(BaseModel):
    items: list[ClassificationItem]
    total: int
    successful: int
    failed: int

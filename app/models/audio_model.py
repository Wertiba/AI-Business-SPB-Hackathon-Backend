from pydantic import BaseModel


class AudioClassificationResponse(BaseModel):
    result: bool
    message: str

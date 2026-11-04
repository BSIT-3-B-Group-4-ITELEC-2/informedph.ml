from datetime import datetime
from pydantic import BaseModel


class ProcessedNews(BaseModel):
    id: str
    title: str
    summary: str
    explanation: str
    published_at: datetime | None


class NewsLLMPipelineResponse(BaseModel):
    results: list[ProcessedNews]

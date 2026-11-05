from datetime import datetime
from pydantic import BaseModel, HttpUrl


class ProcessedNews(BaseModel):
    id: str
    title: str
    summary: str
    explanation: str
    published_at: datetime | None
    url: HttpUrl


class NewsLLMPipelineResponse(BaseModel):
    results: list[ProcessedNews]

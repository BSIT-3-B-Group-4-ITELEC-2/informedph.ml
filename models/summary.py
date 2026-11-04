from pydantic import BaseModel


class SummarizationResult(BaseModel):
    id: str
    title: str
    summary: str
    explanation: str


class SummarizationResponse(BaseModel):
    results: list[SummarizationResult]

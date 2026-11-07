from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import Provide, inject
from containers.container import Container
from pipelines.news_ingestion_pipeline import NewsIngestionPipeline
from pipelines.news_llm_pipeline import NewsLLMPipeline

router = APIRouter(prefix="/news", tags=["News"])


@router.get("/")
@inject
async def get_articles(
    ingestion_pipeline: Annotated[
        NewsIngestionPipeline,
        Depends(Provide[Container.news_ingestion_pipeline]),
    ],
    llm_pipeline: Annotated[
        NewsLLMPipeline,
        Depends(Provide[Container.news_llm_pipeline]),
    ],
):
    try:
        articles = await ingestion_pipeline.execute()

        print(
            f"News ingestion pipeline has successfully retrieve {len(articles)} articles"
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500)

    try:
        processed_news = await llm_pipeline.execute(articles)

        print("News LLM Pipeline has successfully processed the articles")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500)

    return processed_news

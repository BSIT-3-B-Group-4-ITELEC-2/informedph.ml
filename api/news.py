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

        print("Article Count: ", len(articles))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to retrieve news articles.",
                "error": str(e),
            },
        )

    try:
        processed_news = await llm_pipeline.execute(articles)

        print("Processed Articles: ", processed_news)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to process news articles using the LLM pipeline.",
                "error": str(e),
            },
        )

    return processed_news

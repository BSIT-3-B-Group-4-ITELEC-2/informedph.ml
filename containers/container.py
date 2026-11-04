from dependency_injector import containers, providers
from scrapers.gma_rss_scraper import GMARssScraper
from scrapers.philstar_rss_scraper import PhilstarRssScraper
from scrapers.rappler_rss_scraper import RapplerRssScraper
from scrapers.gma_nation_webpage_scraper import GMANationArchiveWebPageScraper
from pipelines.news_ingestion_pipeline import NewsIngestionPipeline
from dotenv import load_dotenv
from clients.create_gemini_client import create_gemini_client
import os
from llm.news_classifier import NewsClassifier
from llm.news_summarizer import NewsSummarizer
from pipelines.news_llm_pipeline import NewsLLMPipeline

load_dotenv()


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    config.gemini_api_key.from_value(os.getenv("GEMINI_API_KEY"))

    gma_rss_scraper = providers.Factory(GMARssScraper)
    philstar_rss_scraper = providers.Factory(PhilstarRssScraper)
    rappler_rss_scraper = providers.Factory(RapplerRssScraper)
    gma_nation_archive_webpage_scraper = providers.Factory(
        GMANationArchiveWebPageScraper
    )

    news_ingestion_pipeline = providers.Factory(
        NewsIngestionPipeline,
        gma_rss_scraper=gma_rss_scraper,
        philstar_rss_scraper=philstar_rss_scraper,
        rappler_rss_scraper=rappler_rss_scraper,
        gma_nation_archive_webpage_scraper=gma_nation_archive_webpage_scraper,
    )

    gemini_client = providers.Singleton(
        create_gemini_client,
        api_key=config.gemini_api_key,
    )

    news_classifier = providers.Singleton(
        NewsClassifier,
        client=gemini_client,
    )

    news_summarizer = providers.Singleton(
        NewsSummarizer,
        client=gemini_client,
    )

    news_llm_pipeline = providers.Singleton(
        NewsLLMPipeline,
        classifier=news_classifier,
        summarizer=news_summarizer,
    )

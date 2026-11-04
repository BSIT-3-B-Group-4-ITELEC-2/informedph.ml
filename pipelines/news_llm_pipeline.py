from llm.news_classifier import NewsClassifier
from llm.news_summarizer import NewsSummarizer
from models.classification import ClassificationResponse
from models.summary import SummarizationResponse
from models.news import ProcessedNews, NewsLLMPipelineResponse


class NewsLLMPipeline:

    def __init__(
        self,
        classifier: NewsClassifier,
        summarizer: NewsSummarizer,
    ):
        self._classifier = classifier
        self._summarizer = summarizer

    async def execute(
        self,
        articles: list[dict],
    ) -> NewsLLMPipelineResponse:

        if not articles:
            return NewsLLMPipelineResponse(results=[])

        prepared_articles = self.__assign_article_ids(articles)

        classification_results = await self._classifier.classify(prepared_articles)

        major_articles = self.__select_major_articles(
            prepared_articles,
            classification_results,
        )

        if not major_articles:
            return NewsLLMPipelineResponse(results=[])

        summarization_results = await self._summarizer.summarize(major_articles)

        return self.__map_published_dates(prepared_articles, summarization_results)

    def __assign_article_ids(
        self,
        articles: list[dict],
    ) -> list[dict]:

        prepared_articles = []

        for index, article in enumerate(articles, start=1):

            prepared_article = article.copy()
            prepared_article["id"] = f"a{index:03d}"

            prepared_articles.append(prepared_article)

        return prepared_articles

    def __select_major_articles(
        self,
        articles: list[dict],
        response: ClassificationResponse,
    ) -> list[dict]:

        major_ids = {
            result.id for result in response.results if result.category == "MAJOR"
        }

        return [article for article in articles if article["id"] in major_ids]

    def __map_published_dates(
        self,
        original_articles: list[dict],
        summarization_results: SummarizationResponse,
    ) -> NewsLLMPipelineResponse:

        article_lookup = {article["id"]: article for article in original_articles}

        processed_news = []

        for result in summarization_results.results:

            original_article = article_lookup[result.id]

            processed_news.append(
                ProcessedNews(
                    id=result.id,
                    title=result.title,
                    summary=result.summary,
                    explanation=result.explanation,
                    published_at=original_article.get("published_at"),
                )
            )

        return NewsLLMPipelineResponse(results=processed_news)

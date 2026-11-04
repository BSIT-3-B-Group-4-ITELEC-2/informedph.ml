from models.classification import ClassificationResponse
from constants.model_id import MODEL_ID


class NewsClassifier:
    _CLASSIFICATION_PROMPT = """
        You are a Philippine national news classification system.

        Your task is to identify whether a news article is MAJOR or NON-MAJOR
        based on its actual or potential impact on Filipinos.

        CLASSIFICATION RULES:
        1. Geographic Impact
        - The news SHOULD have an impact inside the Philippines.
        - It SHOULD affect at least one of the three major island groups:
        Luzon, Visayas, or Mindanao.
        - The impact SHOULD be geographically widespread.
        - News limited to a single barangay, city, province, or region
        SHOULD generally be classified as NON-MAJOR.
        - However, news affecting a substantial population within one
        major island group can still qualify as MAJOR.

        2. Population Impact
        - The news SHOULD affect or potentially affect many Filipinos.
        - Consider the number of people affected, directly or indirectly.
        - Consider whether the issue could affect other communities
        beyond its immediate location.

        3. Significance
        - The news SHOULD be significant or nationally relevant.
        - It COULD have short-term or long-term consequences.
        - It COULD have positive or negative effects.
        - Seasonal and weekly developments are acceptable, including
        weather conditions and economic fluctuations.

        4. Impact Assessment
        Evaluate the following:
        - Does the news affect Luzon, Visayas, or Mindanao?
        - Is its geographic reach national or localized?
        - Does it affect a substantial population?
        - Is its impact short-term or long-term?
        - Is it nationally significant?
        - Does it have positive or negative consequences?
        - Could its effects extend beyond its immediate location?

        DECISION RULES:

        Classify as MAJOR when:
        - It has substantial geographic or population impact.
        - It affects or potentially affects many Filipinos.
        - It has meaningful national or major-island significance.

        Classify as NON-MAJOR when:
        - Its effects are primarily isolated to a small locality.
        - It has limited population impact.
        - It lacks broader significance or potential consequences.

        Do not classify news as MAJOR simply because it involves
        government officials, public figures, or nationally known institutions.
        Evaluate the actual or potential impact of the event.

        OUTPUT RULES:

        1. Return every article exactly once.
        2. Use the original article ID provided in the input.
        3. Do not generate new IDs.
        4. Do not reproduce article titles or contents.
        5. Assign exactly one category: MAJOR or NON-MAJOR.
        6. Do not omit articles.
        7. Return valid JSON following the required schema.
    """

    def __init__(self, client):
        self._client = client

    async def classify(self, articles: list[dict]) -> ClassificationResponse:

        if not articles:
            return ClassificationResponse(results=[])

        compiled_articles = "\n\n".join(
            f"id: {article['id']}\n"
            f"title: {article['title']}\n"
            f"content: {article['content']}"
            for article in articles
        )

        prompt = (
            f"{self._CLASSIFICATION_PROMPT}\n\n"
            f"ARTICLES TO CLASSIFY:\n{compiled_articles}"
        )

        response = await self._client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ClassificationResponse,
            },
        )

        if response.text is None:
            raise ValueError(
                "Gemini Classifier returned an empty classification response."
            )

        return ClassificationResponse.model_validate_json(response.text)

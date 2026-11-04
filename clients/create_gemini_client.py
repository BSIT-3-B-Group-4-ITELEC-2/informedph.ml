from google import genai


def create_gemini_client(api_key: str) -> genai.Client:

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    return genai.Client(api_key=api_key)

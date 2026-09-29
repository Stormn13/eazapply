from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pydantic import SecretStr

load_dotenv()

LLM7_MODEL = "default"
LLM7_BASE_URL = "https://api.llm7.io/v1"


def _create_llm7_model(
    api_key: str,
    request_timeout: float | None = None,
    max_retries: int | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM7_MODEL,
        api_key=SecretStr(api_key),
        base_url=LLM7_BASE_URL,
        use_responses_api=False,
        timeout=request_timeout,
        max_retries=max_retries,
    )


geniric_model = _create_llm7_model(os.environ.get("LLM7_API_KEY", ""))


def create_ingestion_model(timeout_seconds: float) -> ChatOpenAI:
    api_key = os.environ.get("LLM7_API_KEY", "").strip()
    if not api_key:
        raise ValueError("LLM7_API_KEY is not configured; check the environment or .env file.")
    return _create_llm7_model(api_key, request_timeout=timeout_seconds, max_retries=0)

googe_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

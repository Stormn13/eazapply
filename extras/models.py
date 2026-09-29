from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pydantic import SecretStr

load_dotenv()

geniric_model = ChatOpenAI(
    model="default",
    api_key=SecretStr(os.environ["LLM7_API_KEY"]),
    base_url="https://api.llm7.io/v1",
    use_responses_api=False
)

googe_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

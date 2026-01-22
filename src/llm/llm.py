from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
import os


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the Google Gemini LLM instance.
    """
    dotenv.load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        api_key=api_key,
        max_retries=2,
    )

    return model


import os
import asyncio
from dotenv import load_dotenv
from tavily import TavilyClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage

load_dotenv(override=True)


async def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not openai_key or not tavily_key:
        raise ValueError("Missing API keys in .env")

    print("API credentials loaded successfully.")

    # Test OpenAI through the AutoGen model client.
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        api_key=openai_key
    )

    try:
        response = await model_client.create(
            messages=[
                UserMessage(
                    content="Reply with: OpenAI connection successful",
                    source="user"
                )
            ]
        )

        print("OpenAI Response:", response.content)

    finally:
        await model_client.close()

    # Test Tavily search.
    tavily = TavilyClient(api_key=tavily_key)

    results = tavily.search(
        query="latest AI trends in enterprise ERP",
        search_depth="basic",
        max_results=3
    )

    print("\nTavily Search Results:")

    for result in results.get("results", []):
        print("Title:", result.get("title"))
        print("URL:", result.get("url"))
        print("-" * 40)

    print("\nStep 2 API verification completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

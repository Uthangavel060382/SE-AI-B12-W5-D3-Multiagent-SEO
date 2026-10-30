
import os
import json
import asyncio
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()


async def search_trending_topics(query: str) -> str:
    """
    Search recent web coverage and news for a topic.
    Returns relevant source titles, URLs, and snippets.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return json.dumps({
            "error": "TAVILY_API_KEY is missing."
        })

    client = TavilyClient(api_key=api_key)

    def run_search():
        return client.search(
            query=query,
            topic="news",
            time_range="month",
            search_depth="basic",
            max_results=7
        )

    try:
        response = await asyncio.to_thread(run_search)

        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "published_date": item.get("published_date")
            })

        return json.dumps({
            "query": query,
            "period": "past month",
            "results": results
        }, indent=2)

    except Exception as error:
        return json.dumps({
            "error": str(error),
            "query": query
        })

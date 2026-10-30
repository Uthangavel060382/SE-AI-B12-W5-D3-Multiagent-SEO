
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from autogen_ext.models.openai import (
    OpenAIChatCompletionClient
)

from agents.research_agent import (
    create_research_agent
)

load_dotenv(override=True)


async def main():

    topic = input("Enter SEO research topic: ").strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        research_agent = create_research_agent(
            model_client
        )

        print("\nResearch Agent started...\n")

        result = await research_agent.run(
            task=(
                f"Research the topic: {topic}. "
                "Search recent web sources and prepare "
                "a complete SEO research report."
            )
        )

        final_report = str(result.messages[-1].content)

        print("\n========== SEO RESEARCH ==========\n")
        print(final_report)

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "seo_research.txt"

        output_file.write_text(
            final_report,
            encoding="utf-8"
        )

        print(f"\nReport saved to: {output_file}")

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

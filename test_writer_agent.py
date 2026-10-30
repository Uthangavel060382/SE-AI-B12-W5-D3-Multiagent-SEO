
import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from autogen_ext.models.openai import (
    OpenAIChatCompletionClient
)

from agents.writer_agent import (
    create_writer_agent
)

load_dotenv(override=True)


async def main():

    research_file = Path("outputs/seo_research.txt")

    if not research_file.exists():
        print("ERROR: Research report not found.")
        print("Complete Step 3 first.")
        return

    research_content = research_file.read_text(
        encoding="utf-8"
    ).strip()

    if not research_content:
        print("ERROR: Research report is empty.")
        return

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        writer_agent = create_writer_agent(
            model_client
        )

        print("\nSEO Content Writer Agent started...")
        print("\nReading research report...")
        print("\nGenerating SEO article...\n")

        task = f"""
Create a comprehensive SEO article using the
following Research Agent findings.

Use the recommended topic, keywords, audience,
search intent, and verified sources.

Target article length: 1200-1500 words.

RESEARCH REPORT START
{research_content}
RESEARCH REPORT END

Generate the complete article with metadata,
headings, FAQs, and source references.
"""

        result = await writer_agent.run(task=task)

        if not result.messages:
            raise RuntimeError(
                "Writer Agent returned no messages."
            )

        article = result.messages[-1].content

        if not isinstance(article, str) or not article.strip():
            raise RuntimeError(
                "Writer Agent returned no valid article."
            )

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "seo_article_draft.md"

        output_file.write_text(
            article,
            encoding="utf-8"
        )

        print("\n========== GENERATED ARTICLE ==========\n")
        print(article)

        print("\nArticle saved successfully:")
        print(output_file.resolve())

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

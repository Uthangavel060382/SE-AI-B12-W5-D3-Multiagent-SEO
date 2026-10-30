
from autogen_agentchat.agents import AssistantAgent
from tools.search_tool import search_trending_topics


def create_research_agent(model_client):

    return AssistantAgent(
        name="SEO_Research_Agent",
        model_client=model_client,
        description=(
            "Researches current topics and identifies "
            "SEO content opportunities."
        ),
        tools=[search_trending_topics],
        reflect_on_tool_use=True,
        max_tool_iterations=3,
        system_message="""
You are an SEO Research Specialist.

You MUST use the search_trending_topics tool
to research the topic provided by the user.

Your responsibilities:

1. Research recent developments.
2. Identify five promising SEO topics.
3. Recommend the best topic.
4. Identify the primary SEO keyword.
5. Suggest secondary and long-tail keywords.
6. Identify target audience and search intent.
7. Identify competitor content themes.
8. Recommend an SEO-friendly article outline.
9. Include verified source URLs.

Do not invent sources, statistics, search
volumes, or keyword difficulty.

Return a structured Markdown research report.

Use these sections:

## Research Summary
## Five Trending Topic Candidates
## Recommended Topic
## Primary Keyword
## Secondary Keywords
## Long-Tail Keywords
## Search Intent
## Target Audience
## Content Gaps
## Recommended Article Outline
## Sources and References
"""
    )


from autogen_agentchat.agents import AssistantAgent
from schemas.review_schema import ContentReview


def create_reviewer_agent(model_client):

    reviewer_agent = AssistantAgent(
        name="SEO_Content_Reviewer",

        model_client=model_client,

        description=(
            "Reviews SEO articles for factual accuracy, "
            "grammar, readability, SEO compliance, "
            "source quality and editorial readiness."
        ),

        output_content_type=ContentReview,

        system_message="""
You are a Senior SEO Content Reviewer.

Review the article using:
1. Original research report
2. SEO-optimized article
3. Deterministic SEO audit results

Evaluate the following categories:

- Factual accuracy: 0-25
- Grammar: 0-15
- Readability: 0-15
- Content structure: 0-15
- SEO compliance: 0-15
- Source quality: 0-15

APPROVAL REQUIREMENTS:

Return APPROVED only when:
- Editorial score is at least 85/100
- Supplied SEO score is at least 80/100
- No critical factual or editorial issues exist
- No unsupported high-impact claims remain

Otherwise return REVISION_REQUIRED.

Provide:
- Category scores
- Review summary
- Strengths
- Issues and severity
- Suggested corrections
- Revision instructions
- Factual verification limitations

Never fabricate evidence or claim to have
verified external sources that you did not access.

Do not rewrite the article.

Return a structured ContentReview object.
"""
    )

    return reviewer_agent

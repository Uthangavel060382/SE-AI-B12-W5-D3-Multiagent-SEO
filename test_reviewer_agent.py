
import os
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from autogen_ext.models.openai import (
    OpenAIChatCompletionClient
)

from agents.reviewer_agent import (
    create_reviewer_agent
)
from schemas.review_schema import ContentReview

load_dotenv(override=True)

OUTPUT_DIR = Path("outputs")


def read_required_file(filename):
    path = OUTPUT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    content = path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(f"Empty file: {path}")

    return content


def enforce_approval_policy(review, seo_score):
    """
    Apply approval rules in Python, rather than
    relying only on the model's decision.
    """
    scores = review.scores

    editorial_total = sum([
        scores.factual_accuracy,
        scores.grammar,
        scores.readability,
        scores.structure,
        scores.seo_compliance,
        scores.source_quality
    ])

    critical_issues = [
        issue for issue in review.issues
        if issue.severity == "critical"
    ]

    approved = (
        review.decision == "APPROVED"
        and editorial_total >= 85
        and seo_score >= 80
        and len(critical_issues) == 0
    )

    return {
        "decision": (
            "APPROVED"
            if approved else "REVISION_REQUIRED"
        ),
        "editorial_score": editorial_total,
        "seo_score": seo_score,
        "critical_issue_count": len(critical_issues),
        "minimum_editorial_score": 85,
        "minimum_seo_score": 80
    }


async def main():
    print("\nStarting Content Reviewer Agent...")

    research = read_required_file(
        "seo_research.txt"
    )

    article = read_required_file(
        "seo_optimized_article.md"
    )

    audit_text = read_required_file(
        "seo_audit_report.json"
    )

    audit = json.loads(audit_text)
    seo_score = int(
        audit["final_audit"]["seo_score"]
    )

    model_client = OpenAIChatCompletionClient(
        model=os.getenv(
            "OPENAI_MODEL", "gpt-4.1-mini"
        ),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        reviewer = create_reviewer_agent(
            model_client
        )

        task = f"""
Perform a complete editorial review.

RESEARCH REPORT:
<research>
{research}
</research>

OPTIMIZED ARTICLE:
<article>
{article}
</article>

SEO SCORE:
{seo_score}/100

SEO AUDIT:
<audit>
{json.dumps(audit, indent=2)}
</audit>

Evaluate accuracy, grammar, readability,
structure, SEO compliance, and citations.

Return a structured editorial review.
"""

        result = await reviewer.run(task=task)

        review = result.messages[-1].content

        if not isinstance(review, ContentReview):
            raise TypeError(
                "Reviewer did not return ContentReview."
            )

        policy = enforce_approval_policy(
            review, seo_score
        )

        report = {
            "review": review.model_dump(),
            "approval_policy": policy
        }

        review_path = (
            OUTPUT_DIR / "content_review.json"
        )

        review_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        feedback_lines = [
            "# Editorial Review Feedback",
            "",
            f"Decision: {policy['decision']}",
            f"Editorial Score: "
            f"{policy['editorial_score']}/100",
            f"SEO Score: {seo_score}/100",
            "",
            "## Review Summary",
            review.summary,
            "",
            "## Required Revisions"
        ]

        if review.revision_instructions:
            feedback_lines.extend(
                f"- {item}"
                for item in review.revision_instructions
            )
        else:
            feedback_lines.append(
                "- No revisions requested."
            )

        feedback_lines.extend([
            "",
            "## Identified Issues"
        ])

        for issue in review.issues:
            feedback_lines.extend([
                f"### {issue.category} "
                f"({issue.severity})",
                issue.description,
                f"Correction: {issue.suggested_correction}",
                ""
            ])

        feedback_lines.append(
            "## Verification Limitations"
        )

        feedback_lines.extend(
            f"- {item}"
            for item in review.factual_verification_limitations
        )

        feedback_path = (
            OUTPUT_DIR / "review_feedback.md"
        )

        feedback_path.write_text(
            "\n".join(feedback_lines),
            encoding="utf-8"
        )

        print("\n========== REVIEW RESULT ==========")

        print(
            "Editorial Score:",
            policy["editorial_score"],
            "/100"
        )

        print("SEO Score:", seo_score, "/100")
        print("Decision:", policy["decision"])
        print("Issues Identified:", len(review.issues))
        print(
            "Critical Issues:",
            policy["critical_issue_count"]
        )

        print("\nReview saved:", review_path)
        print("Feedback saved:", feedback_path)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

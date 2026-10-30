
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.research_agent import create_research_agent
from agents.writer_agent import create_writer_agent
from agents.seo_agent import create_seo_agent
from agents.reviewer_agent import create_reviewer_agent

from schemas.review_schema import ContentReview
from tools.seo_analyzer import analyze_seo, get_metadata


def save_text(path, content):
    Path(path).write_text(content, encoding="utf-8")


def save_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def extract_text(result):
    content = result.messages[-1].content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Agent returned no text.")
    return content


def approval_policy(review, seo_score):
    scores = review.scores

    editorial = sum([
        scores.factual_accuracy,
        scores.grammar,
        scores.readability,
        scores.structure,
        scores.seo_compliance,
        scores.source_quality
    ])

    critical = sum(
        issue.severity == "critical"
        for issue in review.issues
    )

    passed = (
        review.decision == "APPROVED"
        and editorial >= int(os.getenv("MIN_EDITORIAL_SCORE", 85))
        and seo_score >= int(os.getenv("MIN_SEO_SCORE", 80))
        and critical == 0
    )

    return {
        "decision": "APPROVED" if passed else "REVISION_REQUIRED",
        "editorial_score": editorial,
        "seo_score": seo_score,
        "critical_issues": critical
    }


class SEOWorkflow:

    def __init__(self, topic, run_id=None):
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("Topic must not be empty.")

        self.run_id = run_id or uuid4().hex[:12]
        self.run_dir = Path("outputs") / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.max_revisions = max(
            0, int(os.getenv("MAX_REVIEW_ROUNDS", "2"))
        )

        self.state_file = self.run_dir / "workflow_state.json"
        self.state = {
            "run_id": self.run_id,
            "topic": self.topic,
            "status": "INITIALIZED",
            "current_stage": "INITIALIZED",
            "revision_count": 0,
            "events": [],
            "created_at": utc_now()
        }

        self.model_client = OpenAIChatCompletionClient(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.researcher = create_research_agent(self.model_client)
        self.writer = create_writer_agent(self.model_client)
        self.seo = create_seo_agent(self.model_client)
        self.reviewer = create_reviewer_agent(self.model_client)

        self.persist()

    def persist(self):
        self.state["updated_at"] = utc_now()
        save_json(self.state_file, self.state)

    def log(self, stage, message):
        event = {
            "timestamp": utc_now(),
            "stage": stage,
            "message": message
        }
        self.state["current_stage"] = stage
        self.state["events"].append(event)
        self.persist()
        print(f"\n[{stage}] {message}")

    async def correct_metadata(
        self, article, primary_keyword
    ):
        """
        Ask the SEO Agent to repair only metadata
        if its character limits are invalid.
        """

        audit = analyze_seo(article, primary_keyword)

        if (
            "Meta description length"
            not in audit["recommendations"]
            and "Meta title length"
            not in audit["recommendations"]
        ):
            return article

        self.log(
            "METADATA",
            "Correcting SEO metadata lengths"
        )

        # Request metadata only, not a full rewrite.
        result = await self.seo.run(
            task=(
                "Return exactly two lines and nothing else:\n"
                "SEO Meta Title: <50-60 characters>\n"
                "Meta Description: <140-160 characters>\n\n"
                "Ensure the title includes the primary keyword "
                "naturally. Keep both accurate to the article.\n\n"
                f"PRIMARY KEYWORD: {primary_keyword}\n\n"
                f"ARTICLE:\n<article>\n{article}\n</article>"
            )
        )

        response = extract_text(result)

        import re

        replacements = {}

        for field, minimum, maximum in [
            ("SEO Meta Title", 50, 60),
            ("Meta Description", 140, 160)
        ]:
            value = get_metadata(response, field)

            if minimum <= len(value) <= maximum:
                replacements[field] = value

        updated = article

        for field, value in replacements.items():
            pattern = (
                r"(?im)^(\s*[-*]?\s*"
                + re.escape(field)
                + r"\s*:\s*).+$"
            )

            updated, count = re.subn(
                pattern,
                lambda match: match.group(1) + value,
                updated,
                count=1
            )

            if count != 1:
                raise ValueError(
                    f"Could not replace {field}"
                )

        return updated

  
    
    async def checkpoint(self, stage):
        """
        Save all four AutoGen agent states,
        capture Pydantic warnings, and persist
        workflow checkpoint information.
        """
        import warnings

        states = {}

        agents = [
            ("research", self.researcher),
            ("writer", self.writer),
            ("seo", self.seo),
            ("reviewer", self.reviewer)
        ]

        for name, agent in agents:
            print(f"[CHECKPOINT] Saving {name} agent")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                state = await agent.save_state()

                # Validate JSON serializability.
                json.dumps(state)

            states[name] = state

            if caught:
                for warning in caught:
                    print(
                        f"[WARNING] Agent: {name} | "
                        f"{warning.category.__name__}: "
                        f"{warning.message}"
                    )
            else:
                print(
                    f"[CHECKPOINT] {name}: Saved successfully"
                )

        save_json(
            self.run_dir / "agent_states.json",
            states
        )

        self.state["last_completed_stage"] = stage
        self.persist()


        # Save AutoGen conversational state at stage boundaries.
        states = {}
        for name, agent in [
            ("research", self.researcher),
            ("writer", self.writer),
            ("seo", self.seo),
            ("reviewer", self.reviewer)
        ]:
            states[name] = await agent.save_state()

        save_json(
            self.run_dir / "agent_states.json",
            states
        )
        self.state["last_completed_stage"] = stage
        self.persist()

    async def run(self):
        try:
            self.state["status"] = "RUNNING"
            self.persist()

            # AGENT 1: Research
            self.log("RESEARCH", "Researching user topic")

            research_result = await self.researcher.run(
                task=(
                    f"Research the topic '{self.topic}'. "
                    "Use Tavily search and produce a sourced "
                    "SEO research report."
                )
            )

            research = extract_text(research_result)
            save_text(self.run_dir / "research.md", research)
            await self.checkpoint("RESEARCH")

            # AGENT 2: Initial writing
            self.log("WRITING", "Creating initial article")

            writing_result = await self.writer.run(
                task=(
                    f"Original topic: {self.topic}\n\n"
                    "Create a complete 1200-1500 word SEO "
                    "article using this research report. "
                    "Include the required metadata and "
                    "preserve verified source references.\n\n"
                    f"<research>\n{research}\n</research>"
                )
            )

            article = extract_text(writing_result)
            save_text(self.run_dir / "draft.md", article)
            await self.checkpoint("WRITING")

            primary_keyword = get_metadata(
                article, "Primary Keyword"
            )

            if not primary_keyword:
                raise ValueError(
                    "Writer output lacks Primary Keyword metadata."
                )

            # SEO + REVIEW loop
            for revision in range(self.max_revisions + 1):

                self.state["revision_count"] = revision
                self.persist()

                self.log(
                    "SEO",
                    f"Optimizing article, iteration {revision}"
                )

                initial_audit = analyze_seo(
                    article, primary_keyword
                )

                seo_result = await self.seo.run(
                    task=(
                        "Optimize this article using the "
                        "deterministic SEO audit below. "
                        "Return the complete revised Markdown "
                        "article and preserve source references.\n\n"
                        f"PRIMARY KEYWORD: {primary_keyword}\n\n"
                        f"AUDIT:\n{json.dumps(initial_audit)}\n\n"
                        f"<article>\n{article}\n</article>"
                    )
                )

               
                optimized = extract_text(seo_result)

                # Automatically correct SEO metadata.
                optimized = await self.correct_metadata(
                    optimized,
                    primary_keyword
                )

                # Always recalculate after corrections.
                final_audit = analyze_seo(
                    optimized,
                    primary_keyword
                )


                save_text(
                    self.run_dir / f"optimized_{revision}.md",
                    optimized
                )

                save_json(
                    self.run_dir / f"seo_audit_{revision}.json",
                    {
                        "before": initial_audit,
                        "after": final_audit
                    }
                )

                await self.checkpoint("SEO")

                # AGENT 4: Editorial review
                self.log(
                    "REVIEW",
                    f"Reviewing article, iteration {revision}"
                )

                review_result = await self.reviewer.run(
                    task=(
                        f"Review the optimized article for topic: "
                        f"{self.topic}.\n\n"
                        f"SEO AUDIT:\n{json.dumps(final_audit)}\n\n"
                        f"RESEARCH:\n<research>\n{research}\n"
                        f"</research>\n\n"
                        f"ARTICLE:\n<article>\n{optimized}\n"
                        f"</article>\n\n"
                        "Return the structured editorial review."
                    )
                )

                review = review_result.messages[-1].content

                if not isinstance(review, ContentReview):
                    raise TypeError(
                        "Reviewer returned an invalid response."
                    )

                decision = approval_policy(
                    review, final_audit["seo_score"]
                )

                save_json(
                    self.run_dir / f"review_{revision}.json",
                    {
                        "review": review.model_dump(),
                        "policy": decision
                    }
                )

                await self.checkpoint("REVIEW")

                self.log(
                    "DECISION",
                    f"{decision['decision']} | "
                    f"Editorial: {decision['editorial_score']} | "
                    f"SEO: {decision['seo_score']}"
                )

                if decision["decision"] == "APPROVED":
                    save_text(
                        self.run_dir / "final_article.md",
                        optimized
                    )

                    self.state["status"] = "APPROVED"
                    self.persist()
                    self.log(
                        "COMPLETE",
                        "Article approved by automated review; "
                        "human publication approval is pending."
                    )
                    return self.state

                if revision >= self.max_revisions:
                    self.state["status"] = "REVISION_LIMIT_REACHED"
                    self.persist()
                    self.log(
                        "STOPPED",
                        "Maximum revisions reached; "
                        "manual review required."
                    )
                    return self.state

                # Conditional route back to Writer
                self.log(
                    "REVISION",
                    "Sending review feedback to Content Writer"
                )

                feedback = {
                    "issues": [
                        issue.model_dump()
                        for issue in review.issues
                    ],
                    "revision_instructions":
                        review.revision_instructions,
                    "review_summary": review.summary
                }

                revised_result = await self.writer.run(
                    task=(
                        "Revise the COMPLETE article below using "
                        "the editor's feedback. Preserve the "
                        "original topic, primary keyword, "
                        "metadata format, and verified sources. "
                        "Do not fabricate facts or citations.\n\n"
                        f"FEEDBACK:\n{json.dumps(feedback)}\n\n"
                        f"ARTICLE:\n<article>\n{optimized}\n"
                        "</article>"
                    )
                )

                article = extract_text(revised_result)

                save_text(
                    self.run_dir / f"revised_draft_{revision + 1}.md",
                    article
                )
                await self.checkpoint("REVISION")

        except Exception as error:
            self.state["status"] = "FAILED"
            self.state["error"] = str(error)
            self.persist()
            self.log("ERROR", str(error))
            raise

        finally:
            await self.model_client.close()

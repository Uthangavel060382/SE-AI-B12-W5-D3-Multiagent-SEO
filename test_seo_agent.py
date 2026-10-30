<CodeBlock language="python" editable>
import os
import json
import asyncio
from pathlib import Path


from dotenv import load_dotenv

from autogen_ext.models.openai import (
OpenAIChatCompletionClient
)

from agents.seo_agent import create_seo_agent

from tools.seo_analyzer import (
analyze_seo,
get_metadata
)

load_dotenv(override=True)

async def main():

article_path = Path("outputs/seo_article_draft.md")

if not article_path.exists():
print("ERROR: Writer Agent article not found.")
print("Complete Step 4 first.")
return

article = article_path.read_text(
encoding="utf-8"
)

primary_keyword = get_metadata(
article, "Primary Keyword"
)

if not primary_keyword:
primary_keyword = input(
"Enter primary SEO keyword: "
).strip()

if not primary_keyword:
print("ERROR: Primary keyword is missing.")
return

print("\nPerforming initial SEO analysis...")

initial_audit = analyze_seo(
article, primary_keyword
)

print(
"Initial SEO Score:",
initial_audit["seo_score"],
"/ 100"
)

model_client = OpenAIChatCompletionClient(
model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
api_key=os.getenv("OPENAI_API_KEY")
)

try:
seo_agent = create_seo_agent(model_client)

task = f"""
Optimize the following article.

PRIMARY KEYWORD:
{primary_keyword}

INITIAL SEO AUDIT:
{json.dumps(initial_audit, indent=2)}

ORIGINAL ARTICLE:
{article}

Return the complete optimized Markdown article.
"""

print("\nSEO Specialist Agent started...")

result = await seo_agent.run(task=task)

optimized_article = result.messages[-1].content

if (
not isinstance(optimized_article, str)
or not optimized_article.strip()
):
raise RuntimeError(
"SEO Agent returned an invalid article."
)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

optimized_path = (
output_dir / "seo_optimized_article.md"
)

optimized_path.write_text(
optimized_article,
encoding="utf-8"
)

final_audit = analyze_seo(
optimized_article,
primary_keyword
)

report = {
"initial_audit": initial_audit,
"final_audit": final_audit,
"score_change": (
final_audit["seo_score"]
- initial_audit["seo_score"]
)
}

report_path = (
output_dir / "seo_audit_report.json"
)

report_path.write_text(
json.dumps(report, indent=2),
encoding="utf-8"
)

print("\n========== SEO RESULTS ==========")

print(
"Initial Score:",
initial_audit["seo_score"]
)

print(
"Final Score:",
final_audit["seo_score"]
)

print(
"Score Change:",
report["score_change"]
)

print("\nRemaining SEO recommendations:")

for issue in final_audit["recommendations"]:
print("-", issue)

print("\nOptimized article:", optimized_path)
print("Audit report:", report_path)

finally:
await model_client.close()

if name == "main":
asyncio.run(main())
</CodeBlock>
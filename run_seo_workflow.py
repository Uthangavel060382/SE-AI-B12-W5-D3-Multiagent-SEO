
import asyncio
import json

from dotenv import load_dotenv
from workflows.seo_workflow import SEOWorkflow

load_dotenv(override=True)


async def main():
    print("=" * 55)
    print("AUTOGEN SEO MULTI-AGENT SYSTEM")
    print("=" * 55)

    topic = input("\nEnter your SEO topic: ").strip()

    if not topic:
        print("ERROR: Topic cannot be empty.")
        return

    workflow = SEOWorkflow(topic=topic)

    print("\nWorkflow ID:", workflow.run_id)
    print("Output directory:", workflow.run_dir)

    result = await workflow.run()

    print("\n========== FINAL WORKFLOW RESULT ==========")
    print("Topic:", result["topic"])
    print("Status:", result["status"])
    print("Revision Count:", result["revision_count"])
    print("Last Stage:", result["current_stage"])

    print("\nAll artifacts saved to:")
    print(workflow.run_dir.resolve())

    print("\nWorkflow Summary:")
    print(json.dumps({
        "run_id": result["run_id"],
        "status": result["status"],
        "revision_count": result["revision_count"]
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

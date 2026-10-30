
import json
from pathlib import Path

run_dir = Path("outputs/00c99fb1cbc7")

for iteration in range(3):
    file = run_dir / f"seo_audit_{iteration}.json"

    if not file.exists():
        print(f"Missing: {file}")
        continue

    data = json.loads(file.read_text(encoding="utf-8"))
    audit = data["after"]

    print(f"\n{'=' * 50}")
    print(f"SEO ITERATION {iteration}")
    print(f"{'=' * 50}")

    print("Score:", audit["seo_score"], "/100")
    print("Word count:", audit["word_count"])
    print("Primary keyword:", audit["primary_keyword"])
    print("Keyword occurrences:", audit["keyword_occurrences"])
    print("Keyword density:", audit["keyword_density_percent"])
    print("Meta title length:", audit["meta_title_length"])
    print("Meta description length:", audit["meta_description_length"])

    print("\nSEO CHECK RESULTS:")

    for name, points in audit["checks"].items():
        status = "PASS" if points > 0 else "FAIL"
        print(f"{status:4} | {points:2} points | {name}")

    print("\nREMAINING ISSUES:")
    for item in audit["recommendations"]:
        print("-", item)

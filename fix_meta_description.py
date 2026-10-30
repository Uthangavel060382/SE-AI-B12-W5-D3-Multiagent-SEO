
import re
from pathlib import Path
from tools.seo_analyzer import analyze_seo, get_metadata

run_dir = Path("outputs/00c99fb1cbc7")

source_file = run_dir / "optimized_2.md"
output_file = run_dir / "metadata_corrected.md"

article = source_file.read_text(encoding="utf-8")

primary_keyword = get_metadata(
    article, "Primary Keyword"
)

if not primary_keyword:
    raise ValueError("Primary Keyword metadata not found.")

old_description = get_metadata(
    article, "Meta Description"
)

print("Original description:", old_description)
print("Original length:", len(old_description))

# Proposed replacement for this specific SAP ERP article.
new_description = (
    "Explore Generative AI in SAP ERP, including practical "
    "use cases, business benefits, implementation challenges, "
    "and strategies for enterprise adoption."
)

print("\nNew description:", new_description)
print("New length:", len(new_description))

if not 140 <= len(new_description) <= 160:
    raise ValueError(
        "Meta description must be 140-160 characters."
    )

pattern = (
    r"(?im)^(\s*[-*]?\s*Meta Description\s*:\s*).+$"
)

updated, count = re.subn(
    pattern,
    lambda match: match.group(1) + new_description,
    article,
    count=1
)

if count != 1:
    raise ValueError(
        "Meta Description field not found or not replaced."
    )

output_file.write_text(
    updated,
    encoding="utf-8"
)

result = analyze_seo(
    updated,
    primary_keyword
)

print("\n========== UPDATED SEO REPORT ==========")
print("SEO Score:", result["seo_score"], "/100")
print("Keyword Frequency:", result["keyword_density_percent"])
print("Meta Description Length:", result["meta_description_length"])
print("Remaining Issues:", result["recommendations"])
print("Updated article:", output_file)


import re
import json


def get_metadata(text, field):
    pattern = (
        r"(?im)^\s*(?:[-*]\s*)?"
        + re.escape(field)
        + r"\s*:\s*(.+?)\s*$"
    )

    match = re.search(pattern, text)

    if not match:
        return ""

    return match.group(1).strip().strip("*`")


def analyze_seo(article, primary_keyword):
    """
    Analyze a Markdown SEO article.
    Returns a deterministic score out of 100.
    """

    keyword = primary_keyword.strip().lower()

    if not keyword:
        raise ValueError("Primary keyword is required.")

    meta_title = get_metadata(article, "SEO Meta Title")
    meta_description = get_metadata(
        article, "Meta Description"
    )

    # Isolate article body from metadata.
    body_match = re.search(
        r"(?im)^## Article\s*$",
        article
    )

    body = (
        article[body_match.end():]
        if body_match else article
    )

    # Exclude references from body analysis.
    body = re.split(
        r"(?im)^## References\s*$",
        body
    )[0]

    h1 = re.findall(r"(?m)^# (.+)$", body)
    h2 = re.findall(r"(?m)^## (.+)$", body)
    h3 = re.findall(r"(?m)^### (.+)$", body)

    # Remove Markdown heading markers.
    cleaned = re.sub(
        r"(?m)^#{1,6}\s+", "", body
    )

    words = re.findall(
        r"\b[\w'-]+\b",
        cleaned
    )

    word_count = len(words)

    # Count occurrences of complete primary keyword phrase.
    keyword_count = len(
        re.findall(
            r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
            cleaned.lower()
        )
    )

    # Complete keyword phrase occurrences per 100 words.
    density = (
        keyword_count / max(word_count, 1) * 100
    )

    # First 150 words represent the introduction.
    intro = " ".join(words[:150]).lower()

    faq_present = bool(
        re.search(
            r"(?im)^## (?:Frequently Asked Questions|FAQs)\b",
            body
        )
    )

    reference_present = bool(
        re.search(
            r"(?im)^## References\s*$",
            article
        )
        and re.search(r"https?://", article)
    )

    checks = {
        "Meta title length": (
            15 if 50 <= len(meta_title) <= 60 else 0
        ),
        "Meta description length": (
            15 if 140 <= len(meta_description) <= 160 else 0
        ),
        "Keyword in title and H1": (
            20 if (
                keyword in meta_title.lower()
                and len(h1) == 1
                and keyword in h1[0].lower()
            ) else 0
        ),
        "Keyword in introduction": (
            10 if keyword in intro else 0
        ),
        "Heading structure": (
            10 if len(h2) >= 3 and len(h3) >= 1 else 0
        ),
        "Article length": (
            10 if 1200 <= word_count <= 1500 else 0
        ),
        "Keyword distribution": (
            10 if (
                keyword_count >= 2
                and 0.3 <= density <= 2.0
            ) else 0
        ),
        "FAQ section": (
            5 if faq_present else 0
        ),
        "References": (
            5 if reference_present else 0
        )
    }

    score = sum(checks.values())

    return {
        "primary_keyword": primary_keyword,
        "seo_score": score,
        "maximum_score": 100,
        "word_count": word_count,
        "keyword_occurrences": keyword_count,
        "keyword_density_percent": round(density, 2),
        "meta_title_length": len(meta_title),
        "meta_description_length": len(meta_description),
        "h1_count": len(h1),
        "h2_count": len(h2),
        "h3_count": len(h3),
        "checks": checks,
        "recommendations": [
            name for name, points in checks.items()
            if points == 0
        ]
    }


def format_seo_report(results):
    return json.dumps(results, indent=2)

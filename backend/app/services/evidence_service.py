
import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List


GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}"
    "&hl=en-US&gl=US&ceid=US:en"
)


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    text = _clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords(text: str) -> set:
    normalized = _normalize_text(text)

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "of", "to", "in", "on", "for", "and", "with",
        "as", "at", "by", "from", "this", "that",
        "has", "have", "had", "will", "would", "could",
        "should", "its", "it", "they", "their", "them",
        "than", "into", "over", "under", "about",
        "after", "before", "during", "according",
        "said", "says", "news", "latest"
    }

    return {
        word
        for word in normalized.split()
        if len(word) > 2 and word not in stop_words
    }


def _numbers(text: str) -> set:
    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?\b",
            _normalize_text(text),
        )
    )


def _build_query(caption: str) -> str:
    query = _clean_text(caption)

    filler_phrases = [
        "this photo shows",
        "this image shows",
        "this photograph shows",
        "a photo of",
        "a photograph of",
        "an image of",
        "photo of",
        "picture of",
        "image of",
        "according to the caption",
        "according to this caption",
    ]

    for phrase in filler_phrases:
        query = re.sub(
            re.escape(phrase),
            "",
            query,
            flags=re.IGNORECASE,
        )

    query = re.sub(r"""["'“”‘’]""", "", query)
    query = re.sub(r"\s+", " ", query).strip()

    words = query.split()

    if len(words) > 16:
        query = " ".join(words[:16])

    return query


def _is_visual_only_caption(caption: str) -> bool:
    normalized = _clean_text(caption).lower()

    visual_patterns = [
        r"\bwearing\b",
        r"\bwith\b",
        r"\bwears\b",
        r"\bstanding\b",
        r"\bsitting\b",
        r"\brunning\b",
        r"\bwalking\b",
        r"\bholding\b",
        r"\blooking\b",
        r"\bsmiling\b",
        r"\bperson\b",
        r"\bman\b",
        r"\bwoman\b",
        r"\bdog\b",
        r"\bcat\b",
    ]

    factual_indicators = [
        r"\bpresident\b",
        r"\bprime minister\b",
        r"\bminister\b",
        r"\bgovernment\b",
        r"\belection\b",
        r"\belected\b",
        r"\barrested\b",
        r"\baccused\b",
        r"\bannounced\b",
        r"\bsaid\b",
        r"\bmet\b",
        r"\bmeeting\b",
        r"\bvisited\b",
        r"\bvisit\b",
        r"\battack\b",
        r"\bprotest\b",
        r"\bwar\b",
        r"\bincident\b",
        r"\bdeath\b",
        r"\bkilled\b",
        r"\bappointed\b",
        r"\bappointed as\b",
        r"\brelease\b",
        r"\breleased\b",
        r"\bagreement\b",
        r"\bdeal\b",
        r"\bprice\b",
        r"\boil\b",
        r"\bdiesel\b",
        r"\bsummit\b",
    ]

    has_visual_pattern = any(
        re.search(pattern, normalized)
        for pattern in visual_patterns
    )

    has_factual_indicator = any(
        re.search(pattern, normalized)
        for pattern in factual_indicators
    )

    return has_visual_pattern and not has_factual_indicator


def _build_alternative_queries(caption: str) -> List[str]:
    if _is_visual_only_caption(caption):
        return []

    base_query = _build_query(caption)

    if not base_query:
        return []

    queries = [
        f'"{base_query}"',
        f"{base_query} news",
    ]

    unique_queries = []

    for query in queries:
        if query not in unique_queries:
            unique_queries.append(query)

    return unique_queries


def _retrieve_google_news(
    query: str,
    max_results: int = 10,
) -> List[dict]:

    encoded_query = urllib.parse.quote_plus(query)

    url = GOOGLE_NEWS_RSS.format(
        query=encoded_query
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/130 Safari/537.36"
            )
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=10,
    ) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)

    results = []

    for item in root.findall("./channel/item"):

        title = item.findtext("title") or ""
        google_link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        description = item.findtext("description") or ""

        source_element = item.find("source")

        source_name = (
            source_element.text
            if source_element is not None
            else "Google News"
        )

        clean_excerpt = _clean_text(description)

        if clean_excerpt.startswith(title):
            clean_excerpt = clean_excerpt[len(title):].strip()

        if not title or not google_link:
            continue

        results.append(
            {
                "source": source_name,
                "title": title,
                "publishedDate": pub_date,
                "retrievedDate": datetime.now(
                    timezone.utc
                ).isoformat(),
                "relation": "insufficient",
                "relevance": 0.0,
                "excerpt": clean_excerpt[:500],
                "url": google_link,
                "search_query": query,
            }
        )

        if len(results) >= max_results:
            break

    return results


def _calculate_retrieval_relevance(
    caption: str,
    item: dict,
) -> float:

    caption_words = _keywords(caption)

    evidence_text = (
        f"{item.get('title', '')} "
        f"{item.get('excerpt', '')} "
        f"{item.get('source', '')}"
    )

    evidence_words = _keywords(evidence_text)

    if not caption_words:
        return 0.0

    overlap = caption_words.intersection(
        evidence_words
    )

    keyword_score = (
        len(overlap) / len(caption_words)
    )

    caption_numbers = _numbers(caption)
    evidence_numbers = _numbers(evidence_text)

    numeric_score = 0.0

    if caption_numbers:
        matching_numbers = (
            caption_numbers.intersection(
                evidence_numbers
            )
        )

        numeric_score = (
            len(matching_numbers)
            / len(caption_numbers)
        )

    title_words = _keywords(
        item.get("title", "")
    )

    title_overlap = (
        caption_words.intersection(title_words)
    )

    title_score = (
        len(title_overlap)
        / len(caption_words)
    )

    if caption_numbers:
        score = (
            0.45 * keyword_score
            + 0.35 * title_score
            + 0.20 * numeric_score
        )
    else:
        score = (
            0.55 * keyword_score
            + 0.45 * title_score
        )

    return round(
        min(max(score, 0.0), 1.0),
        4,
    )


def _deduplicate_results(
    results: List[dict],
) -> List[dict]:

    unique_results = []

    seen_urls = set()
    seen_titles = set()

    for item in results:

        url = (
            item.get("url") or ""
        ).strip()

        title_key = _normalize_text(
            item.get("title", "")
        )

        source_key = _normalize_text(
            item.get("source", "")
        )

        identity_key = (
            f"{source_key}|{title_key}"
        )

        if url and url in seen_urls:
            continue

        if identity_key in seen_titles:
            continue

        if url:
            seen_urls.add(url)

        seen_titles.add(identity_key)

        unique_results.append(item)

    return unique_results


def retrieve_evidence(
    caption: str,
    max_results: int = 5,
) -> List[dict]:

    queries = _build_alternative_queries(
        caption
    )

    # --------------------------------------------------
    # Visual-only captions
    # --------------------------------------------------

    if not queries:
        return [
            {
                "source": "External Evidence",
                "title": "No external evidence required",
                "publishedDate": "",
                "retrievedDate": datetime.now(
                    timezone.utc
                ).isoformat(),
                "relation": "insufficient",
                "relevance": 0.0,
                "excerpt": (
                    "The caption describes primarily "
                    "visual properties that should be "
                    "verified from the image itself."
                ),
                "url": "",
                "search_query": "",
            }
        ]

    # --------------------------------------------------
    # Collect more candidates before ranking.
    # --------------------------------------------------

    collected_results = []

    try:

        for query in queries:

            results = _retrieve_google_news(
                query=query,
                max_results=10,
            )

            collected_results.extend(
                results
            )

    except Exception as exc:

        print(
            f"Evidence retrieval failed: {exc}"
        )

    # --------------------------------------------------
    # Remove duplicate articles.
    # --------------------------------------------------

    collected_results = _deduplicate_results(
        collected_results
    )

    # --------------------------------------------------
    # Calculate retrieval relevance.
    # --------------------------------------------------

    for item in collected_results:

        item["relevance"] = (
            _calculate_retrieval_relevance(
                caption=caption,
                item=item,
            )
        )

    # --------------------------------------------------
    # Rank highest-relevance evidence first.
    # --------------------------------------------------

    collected_results.sort(
        key=lambda item: (
            item.get("relevance", 0.0),
            item.get("title", ""),
        ),
        reverse=True,
    )

    # --------------------------------------------------
    # Return only the best candidates.
    # --------------------------------------------------

    return collected_results[:max_results]


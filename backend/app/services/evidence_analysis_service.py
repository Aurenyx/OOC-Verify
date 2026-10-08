import re
from typing import List


def _normalize(text: str) -> str:
    text = text.lower()

    # Preserve letters, numbers, and whitespace.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _keywords(text: str) -> set:
    normalized = _normalize(text)

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "with",
        "as",
        "at",
        "by",
        "from",
        "this",
        "that",
        "these",
        "those",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "its",
        "it",
        "they",
        "their",
        "them",
        "than",
        "into",
        "over",
        "under",
        "about",
        "after",
        "before",
        "during",
        "according",
        "said",
        "says",
        "say",
        "news",
        "report",
        "reports",
        "reported",
    }

    return {
        word
        for word in normalized.split()
        if len(word) > 2 and word not in stop_words
    }


def _numbers(text: str) -> set:
    """
    Extract numeric tokens.

    Examples:
    '100 million barrels' -> {'100'}
    '10 lakh' -> {'10'}
    '4 percent' -> {'4'}
    """
    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?\b",
            text.lower(),
        )
    )


def _calculate_relevance(
    caption: str,
    evidence_text: str,
) -> float:

    caption_words = _keywords(caption)
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

    if caption_numbers:
        score = (
            0.75 * keyword_score
            + 0.25 * numeric_score
        )
    else:
        score = keyword_score

    return round(
        min(score, 1.0),
        4,
    )


def _contains_any(
    text: str,
    patterns: List[str],
) -> bool:

    return any(
        pattern in text
        for pattern in patterns
    )


def _claim_words(text: str) -> set:
    """
    Extract event/claim-specific words.

    These words are useful because an evidence article
    can support a claim simply by describing the same
    event, even when it does not contain words such as
    'confirmed' or 'verified'.
    """

    claim_patterns = [
        # Crime
        "robbery",
        "robbed",
        "rob",
        "theft",
        "stolen",
        "steal",
        "looted",
        "loot",
        "burglary",
        "burglar",
        "thief",
        "thieves",
        "murder",
        "murdered",
        "killed",
        "assault",
        "attacked",
        "attack",
        "kidnap",
        "kidnapped",
        "fraud",
        "scam",
        "arrest",
        "arrested",
        "fir",
        "raid",
        "raided",
        "seized",

        # Accidents / disasters
        "accident",
        "crash",
        "crashed",
        "collision",
        "derailment",
        "derailed",
        "fire",
        "explosion",
        "blast",
        "flood",
        "earthquake",
        "cyclone",
        "storm",
        "landslide",
        "tsunami",
        "disaster",
        "collapsed",
        "collapse",

        # Politics / government
        "election",
        "elections",
        "elected",
        "voting",
        "vote",
        "protest",
        "protests",
        "rally",
        "government",
        "minister",
        "president",
        "prime minister",
        "mla",
        "mp",
        "parliament",
        "assembly",
        "government",

        # Conflict
        "war",
        "conflict",
        "clash",
        "clashes",
        "attack",
        "attacked",
        "airstrike",
        "bombing",
        "violence",
        "ceasefire",

        # Economy / business
        "price",
        "prices",
        "market",
        "stock",
        "stocks",
        "inflation",
        "revenue",
        "profit",
        "loss",
        "investment",
        "funding",
        "release",
        "released",

        # Public events / announcements
        "announcement",
        "announced",
        "agreement",
        "agreed",
        "decision",
        "summit",
        "meeting",
        "conference",
        "launched",
        "launch",
        "approved",
        "signed",

        # Health
        "outbreak",
        "epidemic",
        "pandemic",
        "virus",
        "infection",
        "infected",
        "hospitalized",
        "hospitalised",
        "vaccination",
        "vaccine",

        # Sports
        "match",
        "tournament",
        "championship",
        "final",
        "won",
        "lost",
        "defeated",
        "victory",
        "defeat",
        "score",
        "scored",
        "goal",
        "medal",
        "record",
        "champion",
        "qualified",
        "eliminated",
    ]

    normalized = _normalize(text)

    return {
        pattern
        for pattern in claim_patterns
        if pattern in normalized
    }


def analyze_evidence(
    caption: str,
    evidence_items: List[dict],
) -> List[dict]:

    if not evidence_items:
        return []

    for item in evidence_items:

        title = item.get("title", "")
        excerpt = item.get("excerpt", "")
        source = item.get("source", "")

        # Keep title separate because it is often the
        # strongest representation of the article's claim.
        evidence_text = (
            f"{title} {excerpt} {source}"
        )

        normalized_evidence = _normalize(
            evidence_text
        )

        normalized_caption = _normalize(
            caption
        )

        relevance = _calculate_relevance(
            caption,
            evidence_text,
        )

        caption_keywords = _keywords(
            caption
        )

        evidence_keywords = _keywords(
            evidence_text
        )

        keyword_overlap = (
            caption_keywords.intersection(
                evidence_keywords
            )
        )

        caption_claim_words = _claim_words(
            caption
        )

        evidence_claim_words = _claim_words(
            evidence_text
        )

        claim_overlap = (
            caption_claim_words.intersection(
                evidence_claim_words
            )
        )

        # --------------------------------------------------
        # Numeric agreement
        # --------------------------------------------------

        caption_numbers = _numbers(
            caption
        )

        evidence_numbers = _numbers(
            evidence_text
        )

        numeric_match = bool(
            caption_numbers
            and caption_numbers.intersection(
                evidence_numbers
            )
        )

        # --------------------------------------------------
        # Contradiction indicators
        # --------------------------------------------------

        contradiction_patterns = [
            "did not",
            "didnt",
            "not true",
            "not correct",
            "not accurate",
            "not meet",
            "no meeting",
            "denied",
            "denies",
            "denied that",
            "false claim",
            "false report",
            "false",
            "incorrect",
            "misleading",
            "debunked",
            "fact check",
            "factcheck",
            "untrue",
            "disputed",
            "refuted",
            "rejected the claim",
            "contradicts the claim",
            "police denied",
            "officials denied",
            "government denied",
        ]

        # --------------------------------------------------
        # Explicit support indicators
        # --------------------------------------------------

        support_patterns = [
            "confirmed",
            "confirmed that",
            "verified",
            "officially",
            "official statement",
            "according to",
            "reported that",
            "reports that",
            "announced",
            "announced that",
            "will release",
            "to release",
            "released",
            "release of",
            "approved",
            "agreed to",
            "plans to",
            "scheduled to",
            "held",
            "took place",
            "occurred",
            "happened",
            "police confirmed",
            "officials confirmed",
        ]

        has_contradiction = _contains_any(
            normalized_evidence,
            contradiction_patterns,
        )

        has_support = _contains_any(
            normalized_evidence,
            support_patterns,
        )

        # --------------------------------------------------
        # Evidence classification
        # --------------------------------------------------

        relation = "insufficient"

        # Strong contradiction always gets priority.
        if (
            relevance >= 0.60
            and has_contradiction
        ):
            relation = "contradicts"

        # Explicit support with strong relevance.
        elif (
            relevance >= 0.60
            and has_support
        ):
            relation = "supports"

        # Same specific event/claim expressed in
        # both caption and article.
        #
        # This fixes the previous failure where an
        # article directly stated the same robbery but
        # did not contain words like "confirmed".
        elif (
            relevance >= 0.70
            and len(claim_overlap) >= 1
            and (
                len(keyword_overlap) >= 2
                or numeric_match
            )
        ):
            relation = "supports"

        # Very strong lexical match can also indicate
        # direct article-level support.
        elif (
            relevance >= 0.85
            and len(keyword_overlap) >= 3
            and not has_contradiction
        ):
            relation = "supports"

        # Very low relevance means the article is
        # unrelated to the claim.
        elif relevance < 0.20:

            relation = "irrelevant"

        else:

            relation = "insufficient"

        item["relation"] = relation
        item["relevance"] = relevance

        # --------------------------------------------------
        # Human-readable analysis
        # --------------------------------------------------

        if relation == "supports":

            if claim_overlap:
                item["analysis_reason"] = (
                    "The retrieved source strongly overlaps "
                    "with the specific event or claim in the "
                    "caption and provides consistent textual "
                    "evidence."
                )
            else:
                item["analysis_reason"] = (
                    "The retrieved source is strongly related "
                    "to the caption and contains textual "
                    "information consistent with the claim."
                )

        elif relation == "contradicts":

            item["analysis_reason"] = (
                "The retrieved source is strongly related "
                "to the caption and contains textual "
                "information that contradicts the claim."
            )

        elif relation == "irrelevant":

            item["analysis_reason"] = (
                "The retrieved source has very little "
                "overlap with the specific caption claim."
            )

        else:

            item["analysis_reason"] = (
                "The retrieved source is topically related "
                "but does not provide sufficient textual "
                "evidence to verify or contradict the "
                "specific claim."
            )

    return evidence_items
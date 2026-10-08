import re
from typing import List, Dict


def _extract_qwen_prediction(qwen_response: str) -> str:
    match = re.search(
        r"Prediction:\s*(Genuine|Misleading)",
        qwen_response,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).capitalize()

    return "Pending"


def _calculate_evidence_scores(
    evidence_items: List[Dict],
):
    """
    Calculate normalized evidence scores.

    Each supporting/contradicting source contributes its
    relevance score. The average is used so duplicate or
    repeated sources cannot artificially increase the score.
    """

    support_relevances = []
    contradiction_relevances = []

    for item in evidence_items:

        relation = item.get(
            "relation",
            "insufficient",
        )

        relevance = float(
            item.get(
                "relevance",
                0.0,
            )
        )

        relevance = max(
            0.0,
            min(
                relevance,
                1.0,
            ),
        )

        if relation == "supports":
            support_relevances.append(
                relevance
            )

        elif relation == "contradicts":
            contradiction_relevances.append(
                relevance
            )

    support_score = (
        sum(support_relevances)
        / len(support_relevances)
        if support_relevances
        else 0.0
    )

    contradiction_score = (
        sum(contradiction_relevances)
        / len(contradiction_relevances)
        if contradiction_relevances
        else 0.0
    )

    return (
        round(support_score, 4),
        round(contradiction_score, 4),
    )


def _is_contextual_claim(caption: str) -> bool:
    """
    Detect captions containing event-specific or contextual claims.

    Contextual claims contain information that usually cannot be
    established from pixels alone, including events, crimes,
    organizations, locations, quantities, dates, announcements,
    outcomes, statistics, and reported statements.

    This lightweight heuristic is an additional decision signal,
    not a complete natural-language understanding system.
    """

    if not caption:
        return False

    text = caption.lower()

    contextual_terms = [
        # ---------------------------------------------------------
        # Organizations / institutions / public figures
        # ---------------------------------------------------------
        "g7",
        "g20",
        "government",
        "president",
        "presidential",
        "prime minister",
        "minister",
        "mla",
        "mp",
        "chief minister",
        "governor",
        "mayor",
        "leader",
        "politician",
        "political party",
        "party leader",
        "united nations",
        "nato",
        "who",
        "world bank",
        "imf",
        "supreme court",
        "high court",
        "court",
        "police",
        "authorities",
        "officials",

        # ---------------------------------------------------------
        # Crime / law enforcement
        # ---------------------------------------------------------
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
        "burglars",
        "thief",
        "thieves",
        "crime",
        "criminal",
        "arrest",
        "arrested",
        "arrests",
        "detained",
        "detention",
        "fir",
        "fir filed",
        "case filed",
        "complaint",
        "lawsuit",
        "legal case",
        "murder",
        "murdered",
        "killed",
        "killing",
        "death",
        "dead",
        "assault",
        "attacked",
        "attack",
        "kidnap",
        "kidnapped",
        "kidnapping",
        "abduction",
        "abducted",
        "fraud",
        "scam",
        "scandal",
        "corruption",
        "bribery",
        "accused",
        "suspect",
        "suspects",
        "convicted",
        "conviction",
        "sentenced",
        "jail",
        "prison",
        "custody",
        "investigation",
        "probe",
        "raid",
        "raided",
        "seized",
        "seizure",

        # ---------------------------------------------------------
        # Politics / elections / government actions
        # ---------------------------------------------------------
        "election",
        "elections",
        "elected",
        "vote",
        "voting",
        "poll",
        "polls",
        "ballot",
        "campaign",
        "campaigning",
        "opposition",
        "ruling party",
        "coalition",
        "parliament",
        "assembly",
        "cabinet",
        "policy",
        "bill passed",
        "law passed",
        "legislation",
        "ordinance",
        "resignation",
        "resigned",
        "appointed",
        "appointment",
        "government order",
        "ban",
        "banned",
        "sanction",
        "sanctions",

        # ---------------------------------------------------------
# Events / announcements / official actions
# ---------------------------------------------------------
"release",
"released",
"announce",
"announced",
"announcement",
"agreement",
"agreed",
"decision",
"decided",
"emergency",
"crisis",
"summit",
"meeting",
"meet",
"met",
"visit",
"visits",
"visited",
"visiting",
"conference",
"deal",
"signed",
"signing",
"launched",
"launch",
"inaugurated",
"inauguration",
"opened",
"closure",
"closed",
"shutdown",
"operation",
"operation launched",
"action taken",
"action against",

        # ---------------------------------------------------------
        # Protest / conflict / violence
        # ---------------------------------------------------------
        "protest",
        "protests",
        "protesting",
        "demonstration",
        "demonstrators",
        "rally",
        "strike",
        "striking",
        "riot",
        "riots",
        "clash",
        "clashes",
        "conflict",
        "war",
        "warfare",
        "military operation",
        "airstrike",
        "missile strike",
        "bombing",
        "explosion",
        "blast",
        "violence",
        "violent",
        "ceasefire",
        "hostilities",

        # ---------------------------------------------------------
        # Accidents / disasters
        # ---------------------------------------------------------
        "accident",
        "crash",
        "crashed",
        "collision",
        "overturned",
        "derailed",
        "derailment",
        "train accident",
        "road accident",
        "fire",
        "fire broke out",
        "burned",
        "burnt",
        "explosion",
        "blast",
        "flood",
        "flooding",
        "earthquake",
        "cyclone",
        "hurricane",
        "storm",
        "landslide",
        "avalanche",
        "tsunami",
        "disaster",
        "natural disaster",
        "damage",
        "damaged",
        "destroyed",
        "destroyed by",
        "collapsed",
        "collapse",
        "evacuated",
        "evacuation",
        "rescue operation",
        "rescue",

        # ---------------------------------------------------------
        # Deaths / casualties / injuries
        # ---------------------------------------------------------
        "died",
        "dies",
        "death toll",
        "casualties",
        "casualty",
        "injured",
        "injuries",
        "wounded",
        "fatalities",
        "fatal",
        "killed",
        "missing",
        "missing persons",
        "survivors",
        "victims",

        # ---------------------------------------------------------
        # Economy / finance / business
        # ---------------------------------------------------------
        "economy",
        "economic",
        "financial",
        "finance",
        "market",
        "stock market",
        "stocks",
        "shares",
        "share price",
        "price rises",
        "price falls",
        "prices rise",
        "prices fall",
        "inflation",
        "interest rate",
        "interest rates",
        "revenue",
        "profit",
        "loss",
        "investment",
        "investor",
        "investors",
        "funding",
        "funded",
        "billion",
        "million",
        "crore",
        "lakh",
        "rupees",
        "rs ",
        "₹",
        "dollar",
        "dollars",
        "tax",
        "taxes",
        "budget",

        # ---------------------------------------------------------
        # Medical / health / public health
        # ---------------------------------------------------------
        "outbreak",
        "epidemic",
        "pandemic",
        "disease outbreak",
        "virus",
        "infection",
        "infected",
        "hospitalized",
        "hospitalised",
        "health officials",
        "health ministry",
        "vaccination",
        "vaccinated",
        "vaccine",
        "medical emergency",
        "health emergency",

        # ---------------------------------------------------------
        # Sports events / results
        # ---------------------------------------------------------
        "match",
        "matches",
        "tournament",
        "championship",
        "final",
        "semi-final",
        "quarter-final",
        "won",
        "lost",
        "defeated",
        "victory",
        "defeat",
        "score",
        "scored",
        "goal",
        "goals",
        "medal",
        "medalist",
        "record",
        "world record",
        "league",
        "qualify",
        "qualified",
        "eliminated",
        "champion",
        "champions",

        # ---------------------------------------------------------
        # Science / technology / space events
        # ---------------------------------------------------------
        "mission",
        "launched into space",
        "rocket launch",
        "space mission",
        "satellite",
        "orbit",
        "landing",
        "landed",
        "discovery",
        "researchers found",
        "study found",
        "study shows",
        "scientists found",
        "scientists say",
        "researchers say",
        "experiment",
        "breakthrough",
        "technology announced",

        # ---------------------------------------------------------
        # Quantities / statistics / measurements
        # ---------------------------------------------------------
        "million",
        "billion",
        "thousand",
        "crore",
        "lakh",
        "percent",
        "%",
        "₹",
        "rs ",
        "km",
        "kilometres",
        "kilometers",
        "hours",
        "days",
        "years",
        "people affected",
        "people killed",
        "people injured",
        "death toll",

        # ---------------------------------------------------------
        # Temporal context
        # ---------------------------------------------------------
        "today",
        "yesterday",
        "tomorrow",
        "tonight",
        "this morning",
        "this afternoon",
        "this evening",
        "last night",
        "last week",
        "last month",
        "last year",
        "this week",
        "this month",
        "this year",
        "2026",
        "2025",
        "2024",

        # ---------------------------------------------------------
        # Reporting / attribution language
        # ---------------------------------------------------------
        "according to",
        "reported",
        "reports",
        "reportedly",
        "officials said",
        "police said",
        "police confirmed",
        "authorities said",
        "government said",
        "minister said",
        "official statement",
        "statement issued",
        "sources said",
        "sources claim",
        "witnesses said",
        "witness said",
        "confirmed",
        "confirmed by",

        # ---------------------------------------------------------
        # Location / event-specific wording
        # ---------------------------------------------------------
        "in india",
        "in assam",
        "in delhi",
        "in mumbai",
        "in nagpur",
        "in nashik",
        "in maharashtra",
        "in kolkata",
        "in bengaluru",
        "in hyderabad",
        "in chennai",
        "in pune",
        "in new york",
        "in london",
        "in washington",
        "in uk",
        "in usa",
        "in china",
        "in russia",
        "in ukraine",
        "at the scene",
        "at the site",
        "at the location",
    ]

    return any(term in text for term in contextual_terms)


def make_final_decision(
    alignment_prediction: str,
    alignment_confidence: float,
    qwen_response: str,
    evidence_items: List[Dict],
    caption: str = "",
):
    """
    Evidence-aware multimodal decision fusion.

    Signals:

    1. Alignment classifier
    2. Qwen visual reasoning
    3. Retrieved external evidence
    4. Contextual-claim detection

    Important OOC principle:

    External evidence can confirm that a claimed event or fact
    exists, but it does not automatically prove that the submitted
    image belongs to that specific event.

    The confidence score is a heuristic decision-confidence score,
    NOT a calibrated probability.
    """

    # ---------------------------------------------------------
    # Extract Qwen prediction
    # ---------------------------------------------------------

    qwen_prediction = _extract_qwen_prediction(
        qwen_response
    )

    # ---------------------------------------------------------
    # Detect contextual/event claim
    # ---------------------------------------------------------

    contextual_claim = _is_contextual_claim(
        caption
    )

    # ---------------------------------------------------------
    # Calculate evidence scores
    # ---------------------------------------------------------

    (
        support_score,
        contradiction_score,
    ) = _calculate_evidence_scores(
        evidence_items
    )

    # ---------------------------------------------------------
    # Evidence state
    # ---------------------------------------------------------

    strong_support = (
        support_score >= 0.70
    )

    strong_contradiction = (
        contradiction_score >= 0.70
    )

    # ---------------------------------------------------------
    # 1. Strong external contradiction
    # ---------------------------------------------------------

    if strong_contradiction:

        final_prediction = "Misleading"

        confidence = (
            0.75
            + (
                0.15
                * contradiction_score
            )
        )

        reason = (
            "Retrieved evidence strongly contradicts "
            "the caption claim."
        )

    # ---------------------------------------------------------
    # 2. Strong alignment mismatch + Qwen Genuine
    #
    # This protects against obvious visual mismatches.
    #
    # Example:
    # Image: fuel pump
    # Caption: person walking through a forest
    #
    # Alignment classifier:
    # Misleading with very high confidence
    #
    # Qwen:
    # Genuine
    #
    # Result:
    # Misleading
    # ---------------------------------------------------------

    elif (
        qwen_prediction == "Genuine"
        and alignment_prediction == "Misleading"
        and alignment_confidence >= 0.90
    ):

        final_prediction = "Misleading"

        confidence = (
            0.65
            + (
                0.20
                * alignment_confidence
            )
        )

        reason = (
            "The multimodal model considered the caption visually "
            "plausible, but the alignment classifier showed a very "
            "strong mismatch between the image and caption."
        )

    # ---------------------------------------------------------
    # 3. Strong external support + Qwen Genuine +
    #    contextual claim + alignment mismatch
    #
    # This is the main OOC protection.
    #
    # Evidence proves that the claimed event exists.
    # It does NOT prove that the submitted image belongs
    # to that event.
    # ---------------------------------------------------------

    elif (
        strong_support
        and qwen_prediction == "Genuine"
        and contextual_claim
        and alignment_prediction == "Misleading"
    ):

        final_prediction = "Misleading"

        confidence = (
            0.65
            + (
                0.20
                * alignment_confidence
            )
        )

        reason = (
            "Retrieved evidence confirms the captioned event, "
            "but the image-caption alignment indicates that the "
            "submitted image may be out of context for the "
            "specific event described."
        )

    # ---------------------------------------------------------
    # 4. Strong external support + Qwen Genuine
    #
    # Normal genuine case.
    # ---------------------------------------------------------

    elif (
        strong_support
        and qwen_prediction == "Genuine"
    ):

        final_prediction = "Genuine"

        confidence = (
            0.75
            + (
                0.15
                * support_score
            )
        )

        reason = (
            "The image-caption relationship is visually "
            "consistent and retrieved evidence strongly "
            "supports the caption."
        )

    # ---------------------------------------------------------
    # 5. Strong external support + Qwen Misleading
    # ---------------------------------------------------------

    elif (
        strong_support
        and qwen_prediction == "Misleading"
        and contextual_claim
        and alignment_prediction == "Genuine"
    ):

        final_prediction = "Genuine"

        confidence = (
            0.65
            + (
                0.20
                * support_score
            )
        )

        reason = (
            "Retrieved evidence strongly supports the "
            "captioned contextual claim, and the image-caption "
            "alignment is consistent. The multimodal model "
            "could not independently establish the full event "
            "from visual content alone, which is expected for "
            "contextual news claims."
        )

    # ---------------------------------------------------------
    # 6. Qwen says Misleading without strong evidence
    # ---------------------------------------------------------

    elif qwen_prediction == "Misleading":

        final_prediction = "Misleading"

        if alignment_prediction == "Misleading":

            confidence = (
                0.60
                + (
                    0.20
                    * alignment_confidence
                )
            )

        else:

            confidence = 0.60

        reason = (
            "The multimodal model found that the image "
            "does not provide sufficient visual evidence "
            "to establish the captioned claim."
        )

    # ---------------------------------------------------------
    # 7. Qwen says Genuine
    # ---------------------------------------------------------

    elif qwen_prediction == "Genuine":

        final_prediction = "Genuine"

        if alignment_prediction == "Genuine":

            confidence = (
                0.60
                + (
                    0.20
                    * alignment_confidence
                )
            )

            reason = (
                "The multimodal model found the image and "
                "caption visually consistent, supported by "
                "the image-caption alignment signal."
            )

        else:

            confidence = 0.60

            reason = (
                "The multimodal model found the image and "
                "caption visually consistent, although the "
                "alignment classifier indicated a potentially "
                "misleading relationship."
            )

    # ---------------------------------------------------------
    # 8. Qwen unavailable
    # ---------------------------------------------------------

    elif alignment_prediction == "Misleading":

        final_prediction = "Misleading"

        confidence = min(
            0.80,
            max(
                0.55,
                alignment_confidence,
            ),
        )

        reason = (
            "The multimodal reasoning result was unavailable, "
            "so the alignment classifier detected a potentially "
            "misleading relationship."
        )

    # ---------------------------------------------------------
    # 9. No strong signal
    # ---------------------------------------------------------

    else:

        final_prediction = "Genuine"

        confidence = min(
            0.80,
            max(
                0.50,
                alignment_confidence,
            ),
        )

        reason = (
            "No strong contradiction was identified by "
            "the available visual and evidence signals."
        )

    # ---------------------------------------------------------
    # Final confidence normalization
    # ---------------------------------------------------------

    confidence = round(
        min(
            confidence,
            0.95,
        ),
        4,
    )

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    explanation = (
        f"Final prediction: {final_prediction}. "
        f"Alignment classifier predicted "
        f"{alignment_prediction} with confidence "
        f"{alignment_confidence:.2f}. "
        f"Qwen visual reasoning predicted "
        f"{qwen_prediction}. "
        f"Contextual claim detected: "
        f"{contextual_claim}. "
        f"Retrieved evidence contained "
        f"{len(evidence_items)} items, with "
        f"{support_score:.2f} support score and "
        f"{contradiction_score:.2f} contradiction score. "
        f"{reason}"
    )

    # ---------------------------------------------------------
    # Structured result
    # ---------------------------------------------------------

    return {
        "prediction": final_prediction,
        "confidence_score": confidence,
        "reason": reason,
        "explanation": explanation,
        "qwen_prediction": qwen_prediction,
        "support_score": support_score,
        "contradiction_score": contradiction_score,
    }
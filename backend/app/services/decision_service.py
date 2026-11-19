import logging
import re
from typing import List, Dict, Optional


logger = logging.getLogger("uvicorn.error")



def extract_qwen_prediction(qwen_response: Optional[str]) -> Optional[str]:
    """
    Extract structured Genuine, Misleading, or Inconclusive from Qwen output.
    Returns None if unavailable, failed, or unparseable.
    """
    if not qwen_response or not isinstance(qwen_response, str):
        return None

    if "mllm analysis failed:" in qwen_response.lower():
        return None

    match = re.search(
        r"Prediction:\s*(Genuine|Misleading|Inconclusive)\b",
        qwen_response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).capitalize()

    return None


def _extract_qwen_prediction(qwen_response: str) -> str:
    return extract_qwen_prediction(qwen_response) or "Pending"



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


def _determine_decision_branch(
    qwen_prediction: str,
    contextual_claim: bool = False,
    strong_support: bool = False,
    strong_contradiction: bool = False,
    alignment_prediction: str = "Pending",
    alignment_confidence: float = 0.0,
) -> str:
    """Determine the exact final-decision branch that will be selected."""
    if qwen_prediction == "Genuine":
        if alignment_prediction == "Genuine":
            return "1a. Qwen Genuine + Alignment Genuine (agreement)"
        if alignment_prediction == "Misleading":
            return "1b. Qwen Genuine + Alignment Misleading (Qwen prioritized)"
        return "1c. Qwen Genuine + Alignment Other/Pending"

    if qwen_prediction == "Misleading":
        if alignment_prediction == "Misleading":
            return "2a. Qwen Misleading + Alignment Misleading (agreement)"
        if alignment_prediction == "Genuine":
            return "2b. Qwen Misleading + Alignment Genuine (Qwen prioritized)"
        return "2c. Qwen Misleading + Alignment Other/Pending"

    if qwen_prediction == "Inconclusive":
        return "3. Qwen Inconclusive -> Inconclusive fallback"

    return "4. Fallback / Qwen unavailable -> Inconclusive fallback"


def make_final_decision(
    alignment_prediction: str,
    alignment_confidence: float,
    qwen_response: str,
    evidence_items: List[Dict],
    caption: str = "",
):
    """
    Evidence-aware multimodal decision fusion.

    Policy:
    1. When Qwen returns a valid parsed prediction of 'Genuine' or 'Misleading',
       that prediction determines the final prediction. Alignment disagreement,
       missing evidence, or fusion heuristics do not override a valid Qwen verdict.
    2. Alignment signals and retrieved evidence provide corroboration and calibrate
       the heuristic decision confidence score (not a calibrated probability).
    3. When Qwen output is inconclusive, unavailable, or unparseable, the decision
       falls back safely to 'Inconclusive' (confidence 0.50).
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

    # Diagnostic logging immediately before final prediction branches execute
    branch_selected = _determine_decision_branch(
        qwen_prediction=qwen_prediction,
        contextual_claim=contextual_claim,
        strong_support=strong_support,
        strong_contradiction=strong_contradiction,
        alignment_prediction=alignment_prediction,
        alignment_confidence=alignment_confidence,
    )

    logger.info(
        "[DECISION-DIAG] Pre-decision signals:\n"
        "  - qwen_prediction: %s\n"
        "  - contextual_claim: %s\n"
        "  - strong_support: %s (support_score=%.4f)\n"
        "  - strong_contradiction: %s (contradiction_score=%.4f)\n"
        "  - alignment_prediction: %s\n"
        "  - alignment_confidence: %.4f\n"
        "  - exact final-decision branch selected: %s",
        qwen_prediction,
        contextual_claim,
        strong_support,
        support_score,
        strong_contradiction,
        contradiction_score,
        alignment_prediction,
        alignment_confidence,
        branch_selected,
    )

    # ---------------------------------------------------------
    # 1. Qwen says Genuine
    # ---------------------------------------------------------

    if qwen_prediction == "Genuine":

        final_prediction = "Genuine"

        if alignment_prediction == "Genuine":

            confidence = min(
                0.85,
                0.65 + (0.20 * alignment_confidence),
            )

            if strong_support:
                confidence = max(
                    confidence,
                    0.75 + (0.15 * support_score),
                )

            reason = (
                "The multimodal model found the image and "
                "caption visually consistent, supported by "
                "the image-caption alignment signal."
            )

        elif alignment_prediction == "Misleading":

            confidence = 0.60

            if strong_support:
                confidence = max(
                    confidence,
                    0.60 + (0.15 * support_score),
                )

            reason = (
                "The multimodal model found the image and "
                "caption visually consistent. The alignment "
                "classifier indicated a potential mismatch, "
                "but visual reasoning confirmed genuine consistency."
            )

        else:

            confidence = 0.60

            if strong_support:
                confidence = max(
                    confidence,
                    0.75 + (0.15 * support_score),
                )

            reason = (
                "The multimodal model found the image and "
                "caption visually consistent."
            )

    # ---------------------------------------------------------
    # 2. Qwen says Misleading
    # ---------------------------------------------------------

    elif qwen_prediction == "Misleading":

        final_prediction = "Misleading"

        if alignment_prediction == "Misleading":

            confidence = min(
                0.85,
                0.65 + (0.20 * alignment_confidence),
            )

            if strong_contradiction:
                confidence = max(
                    confidence,
                    0.75 + (0.15 * contradiction_score),
                )

            reason = (
                "The multimodal model identified a potentially "
                "misleading image-caption relationship, supported by "
                "the image-caption alignment signal."
            )

        elif alignment_prediction == "Genuine":

            confidence = 0.60

            if strong_contradiction:
                confidence = max(
                    confidence,
                    0.60 + (0.15 * contradiction_score),
                )

            reason = (
                "The multimodal model identified a potentially "
                "misleading image-caption relationship. Although "
                "the alignment classifier indicated semantic alignment, "
                "visual reasoning determined the image does not "
                "support the caption."
            )

        else:

            confidence = 0.60

            if strong_contradiction:
                confidence = max(
                    confidence,
                    0.75 + (0.15 * contradiction_score),
                )

            reason = (
                "The multimodal model identified a potentially "
                "misleading image-caption relationship. Review the "
                "visual explanation and available evidence."
            )

    # ---------------------------------------------------------
    # 3. Qwen says Inconclusive
    # ---------------------------------------------------------

    elif qwen_prediction == "Inconclusive":

        final_prediction = "Inconclusive"
        confidence = 0.50

        reason = (
            "The visual model could neither establish nor "
            "contradict the caption using the available image "
            "evidence. Further contextual evidence may be needed."
        )

    # ---------------------------------------------------------
    # 4. Fallback: Qwen unavailable, unparseable, or failed
    # ---------------------------------------------------------

    else:

        final_prediction = "Inconclusive"
        confidence = 0.50

        reason = (
            "Insufficient evidence or unavailable multimodal signals "
            "to conclusively verify the image-caption claim."
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
    # Explanation (substantive MLLM reasoning)
    # ---------------------------------------------------------

    is_qwen_valid = (
        bool(qwen_response)
        and isinstance(qwen_response, str)
        and bool(qwen_response.strip())
        and "mllm analysis failed" not in qwen_response.lower()
    )

    if is_qwen_valid:
        explanation = qwen_response.strip()
    else:
        explanation = "MLLM explanation unavailable"

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
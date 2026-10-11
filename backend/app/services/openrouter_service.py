
import base64
import logging
import os
import re

from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_VISION_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]


def get_candidate_models() -> list[str]:
    """Return prioritized, deduplicated list of multimodal vision models (max 3 for OpenRouter)."""
    configured_model = os.getenv("OPENROUTER_MODEL", "").strip()
    candidates: list[str] = []

    if configured_model and configured_model.lower() != "openrouter/free":
        candidates.append(configured_model)

    for fallback in DEFAULT_VISION_MODELS:
        if fallback not in candidates:
            candidates.append(fallback)

    return candidates[:3]


MODEL_NAME = DEFAULT_VISION_MODELS[0]

CLAIM_UNCERTAINTY_PATTERNS = [
    # 1. Insufficient / not enough / inconclusive evidence / context to verify / determine / confirm / assess / judge / prove
    r"\b(?:(?:visual\s+)?(?:evidence|information|details?|context)\s+is\s+(?:insufficient|not enough|inconclusive)|(?:(?:does not|doesn't|do not)\s+(?:have|provide|contain|offer)\s+(?:sufficient|enough)|insufficient|not enough|lacks?|inconclusive)\s+(?:visual\s+)?(?:evidence|information|details?|context))\b.*?\bto\s+(?:verify|confirm|determine|conclude|establish|assess|judge|substantiate|prove)\b",

    # 2. Neither confirms nor refutes / contradicts / denies / proves nor disproves
    r"\bneither\s+(?:confirms?|verifies|proves|supports?)\s+nor\s+(?:refutes?|contradicts?|denies?|disproves?)\b",
    r"\b(?:cannot|unable to|impossible to|not possible to|too\s+\w+\s+to)\s+(?:confirm|verify|determine)\s+or\s+(?:refute|contradict|deny|disprove)\b",

    # 3. Cannot / unable to / impossible to verify / confirm / substantiate the claim / caption / event / statement / authenticity
    r"\b(?:cannot|can not|can't|unable to|impossible to|not possible to|fails? to|difficult to|hard to)\s+(?:be\s+)?(?:verif(?:y|ied)|confirm(?:ed)?|substantiat(?:e|ed)|ascertain(?:ed)?)\s+(?:the\s+)?(?:overall\s+)?(?:claim|caption|event|statement|assertion|authenticity|context)\b",

    # 4. The claim / caption / event / authenticity cannot be verified / confirmed / determined / substantiated
    r"\b(?:overall\s+)?(?:claim|caption|event|statement|assertion|authenticity)\s+(?:cannot|can not|can't|unable to|is impossible to)\s+(?:be\s+)?(?:verif(?:y|ied)|confirm(?:ed)?|determin(?:e|ed)|substantiat(?:e|ed)|ascertain(?:ed)?)\b",

    # 5. Cannot / unable / impossible to determine / tell / verify whether / if the image supports / depicts / matches the caption / claim
    r"\b(?:cannot|can not|can't|unable to|impossible to|not possible to|fails? to)\s+(?:be\s+)?(?:determin(?:e|ed)|tell|judg(?:e|ed)|conclud(?:e|ed)|said|verif(?:y|ied)|confirm(?:ed)?)\s+(?:whether|if)\s+(?:the\s+)?(?:image\s+)?(?:visually\s+)?(?:supports?|depicts?|matches?|confirms?|contradicts?|aligns with)\s+(?:the\s+)?(?:caption|claim)\b",

    # 6. Cannot / unable / impossible to determine / verify whether / if the claim / caption / event is true / false / genuine / misleading
    r"\b(?:cannot|can not|can't|unable to|impossible to|not possible to|fails? to)\s+(?:be\s+)?(?:determin(?:e|ed)|tell|judg(?:e|ed)|conclud(?:e|ed)|said|verif(?:y|ied)|confirm(?:ed)?)\s+(?:whether|if)\s+(?:the\s+)?(?:claim|caption|event|statement)\s+is\s+(?:true|false|genuine|misleading|accurate|real|authentic)\b",

    # 7. Cannot / unable / impossible to determine whether / if the photo / image depicts the claimed / captioned event / incident
    r"\b(?:cannot|can not|can't|unable to|impossible to|not possible to|fails? to)\s+(?:be\s+)?(?:determin(?:e|ed)|tell|judg(?:e|ed)|conclud(?:e|ed)|said|verif(?:y|ied)|confirm(?:ed)?)\s+(?:whether|if)\s+(?:this|the photo|the image|the scene)\s+(?:is|depicts?|shows?)\s+(?:the\s+)?(?:claimed|captioned|described|stated)\s+(?:event|incident|claim|location|scene)\b",

    # 8. Explicit uncertainty about whether image supports caption or claim is true
    r"\b(?:uncertain|inconclusive|unclear|unverified)\s+(?:as to\s+)?(?:whether|if)\s+(?:the\s+)?(?:image\s+)?(?:visually\s+)?(?:supports?|depicts?|matches?|confirms?|contradicts?)\s+(?:the\s+)?(?:caption|claim)\b",
    r"\b(?:uncertain|inconclusive|unclear|unverified)\s+(?:as to\s+)?(?:whether|if)\s+(?:the\s+)?(?:claim|caption|event)\s+is\s+(?:true|false|genuine|misleading|accurate|real|authentic)\b",

    # 9. Overall claim / caption is inconclusive / unverifiable / unverified
    r"\b(?:overall\s+)?(?:claim|caption|event)\s+is\s+(?:inconclusive|unverifiable|unverified|uncertain)\b",

    # 10. Cannot / impossible to verify / determine from the image alone
    r"\b(?:cannot|unable to|impossible to|not possible to)\s+(?:be\s+)?(?:verif(?:y|ied)|confirm(?:ed)?|determin(?:e|ed))\s+from\s+(?:the\s+)?(?:image|visual evidence)\s+alone\b",
]

CONTRADICTION_PATTERNS = [
    r"\b(?:directly|clearly|explicitly|visually)?\s*(?:contradicts?|contradictory|contradiction)\b",
    r"\b(?:inconsistent with|inconsistency between)\b",
    r"\b(?:mismatch(?:es|ed|ing)?)\b",
    r"\b(?:shows?|depicts?)\s+(?:a\s+)?(?:completely\s+)?(?:different|unrelated)\b",
    r"\binstead of\b",
]


def has_explicit_visual_contradiction(text: str) -> bool:
    """Detect whether text describes explicit visual contradiction/mismatch with the caption."""
    if not text:
        return False
    return any(re.search(pat, text, re.IGNORECASE) for pat in CONTRADICTION_PATTERNS)


def has_claim_level_uncertainty(text: str) -> bool:
    """Detect conservative, claim-level uncertainty in explanation text."""
    if not text:
        return False
    return any(re.search(pat, text, re.IGNORECASE) for pat in CLAIM_UNCERTAINTY_PATTERNS)


def is_compound_prediction(text: str) -> bool:
    """Detect ambiguous compound predictions (e.g., 'Genuine or Inconclusive', 'Genuine/Misleading')."""
    clean_text = text.strip()
    verdict_tokens = re.findall(
        r"\b(genuine|misleading|inconclusive|uncertain|unverified|cannot determine|unable to determine|insufficient (?:visual )?evidence)\b",
        clean_text,
        re.IGNORECASE,
    )
    unique_tokens = {t.lower() for t in verdict_tokens}
    if len(unique_tokens) > 1:
        return True

    if re.search(
        r"\b(?:genuine|misleading|inconclusive|uncertain|unverified)\s*(?:/|\bor\b|\band\b|\bvs\b|\bversus\b)",
        clean_text,
        re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b(?:either|both|neither)\b.*?\b(?:genuine|misleading|inconclusive|uncertain|unverified)\b",
        clean_text,
        re.IGNORECASE,
    ):
        return True

    return False


def normalize_prediction_label(raw: str) -> str:
    raw_lower = raw.lower()
    if raw_lower in {
        "inconclusive",
        "uncertain",
        "unverified",
        "cannot determine",
        "unable to determine",
        "insufficient evidence",
        "insufficient visual evidence",
    }:
        return "Inconclusive"
    elif raw_lower == "genuine":
        return "Genuine"
    elif raw_lower == "misleading":
        return "Misleading"
    return raw_lower.capitalize()


def analyze_remote_image(
    image_bytes: bytes,
    content_type: str,
    caption: str,
) -> dict:
    """Analyze whether an image visually supports its caption."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not configured.")
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    if not image_bytes:
        raise ValueError("Image data is empty.")

    if not caption or not caption.strip():
        raise ValueError("Caption cannot be empty.")

    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        content_type = "image/jpeg"

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{content_type};base64,{encoded_image}"

    prompt = (
        "Analyze whether the image visually supports the caption. "
        "Judge visual consistency only; do not claim external events are verified. "
        "Return exactly these fields on separate lines:\n"
        "Prediction: Genuine, Misleading, or Inconclusive\n"
        "Reason: A short explanation\n"
        "Visual Evidence: Details visible in the image\n\n"
        f"Caption: {caption}"
    )

    candidate_models = get_candidate_models()
    primary_model = candidate_models[0]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "OOC-Verify",
    }
    payload_data = {
        "model": primary_model,
        "models": candidate_models,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 700,
    }

    try:
        logger.debug("Calling OpenRouter with candidate models: %s", candidate_models)
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload_data,
            timeout=90,
        )
        logger.debug("OpenRouter HTTP status: %s", response.status_code)
    except requests.Timeout as exc:
        logger.error("OpenRouter connection timeout: %s", exc)
        raise RuntimeError("Remote model request timed out.") from None
    except requests.ConnectionError as exc:
        logger.error("OpenRouter connection error: %s", exc)
        raise RuntimeError(f"Failed to connect to remote model service: {exc}") from None
    except requests.RequestException as exc:
        logger.error("OpenRouter request error: %s", exc)
        raise RuntimeError(f"Remote model request failed: {exc}") from None

    if response.status_code != 200:
        error_detail = response.text[:1000]
        try:
            err_json = response.json()
            if isinstance(err_json, dict) and "error" in err_json:
                err_val = err_json["error"]
                if isinstance(err_val, dict):
                    error_detail = err_val.get("message", error_detail)
                else:
                    error_detail = str(err_val)
        except Exception:
            pass

        safe_detail = str(error_detail).replace(api_key, "[REDACTED]")
        logger.error("OpenRouter error %s: %s", response.status_code, safe_detail)
        raise RuntimeError(
            f"Remote model request failed with status {response.status_code}: {safe_detail}"
        )

    try:
        payload = response.json()
    except Exception as exc:
        safe_snippet = response.text[:1000].replace(api_key, "[REDACTED]")
        logger.error("OpenRouter invalid JSON: %s. Snippet: %s", exc, safe_snippet)
        raise RuntimeError(f"OpenRouter returned invalid JSON: {exc}") from None

    if not isinstance(payload, dict):
        logger.error("OpenRouter unexpected payload type: %s", type(payload))
        raise RuntimeError("OpenRouter returned an unexpected response format: root is not a dictionary.")

    if "error" in payload:
        error_info = payload["error"]
        msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
        safe_msg = str(msg).replace(api_key, "[REDACTED]")
        logger.error("OpenRouter API error in response payload: %s", safe_msg)
        raise RuntimeError(f"OpenRouter API error: {safe_msg}")

    try:
        choices = payload.get("choices")
        if not choices or not isinstance(choices, list):
            safe_payload = str(payload)[:500].replace(api_key, "[REDACTED]")
            logger.error("OpenRouter response missing choices array.")
            raise RuntimeError("OpenRouter returned no choices in response.")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            logger.error("OpenRouter choice missing message object.")
            raise RuntimeError("OpenRouter returned an unexpected response format.")

        model_output = message.get("content")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("OpenRouter parsing error: %s", exc)
        raise RuntimeError("OpenRouter returned an unexpected response format.") from None

    if not isinstance(model_output, str) or not model_output.strip():
        logger.error("The remote model returned an empty response.")
        raise RuntimeError("The remote model returned an empty response.")

    cleaned = re.sub(r"[*_`#]", "", model_output).strip()

    # Reject content safety responses explicitly
    if re.search(r"(?im)^\s*(?:[\d]+[\.\)]\s*|[-*]\s*)?User Safety\s*:\s*safe\b", cleaned):
        logger.error("Remote model returned content safety classification.")
        raise RuntimeError(
            "The remote model returned a content safety classification instead of a verification prediction."
        )

    # 1. Reject ambiguous compound verdicts on any labeled prediction line (supports numbered/bulleted prefixes)
    labeled_matches = list(
        re.finditer(
            r"(?im)^\s*(?:[\d]+[\.\)]\s*|[-*]\s*)?(?:prediction|verdict|classification|decision)\s*[:\-]\s*(.*)$",
            cleaned,
        )
    )

    for match in labeled_matches:
        line_content = match.group(1).strip()
        if is_compound_prediction(line_content):
            logger.error("Remote model returned ambiguous compound prediction.")
            raise RuntimeError(
                "The remote model did not provide a recognizable prediction."
            )

    # 2. Extract candidate predictions from labeled lines
    labeled_verdicts: list[str] = []
    for match in labeled_matches:
        line_content = match.group(1).strip()
        unc_match = re.search(
            r"\b(inconclusive|uncertain|unverified|cannot determine|unable to determine|insufficient (?:visual )?evidence)\b",
            line_content,
            re.IGNORECASE,
        )
        if unc_match:
            labeled_verdicts.append("Inconclusive")
            continue
        gen_match = re.search(r"\bgenuine\b", line_content, re.IGNORECASE)
        if gen_match:
            labeled_verdicts.append("Genuine")
            continue
        mis_match = re.search(r"\bmisleading\b", line_content, re.IGNORECASE)
        if mis_match:
            labeled_verdicts.append("Misleading")
            continue

    prediction: str | None = None
    unique_labeled = list(dict.fromkeys(labeled_verdicts))

    if len(unique_labeled) > 1:
        # Multiple conflicting labeled prediction lines detected.
        # Do not silently select the first verdict; return Inconclusive safely.
        logger.warning("Multiple conflicting labeled prediction lines detected. Returning Inconclusive.")
        prediction = "Inconclusive"
    elif len(unique_labeled) == 1:
        prediction = unique_labeled[0]

    # 3. Fallback: accept a clear standalone prediction line if no labeled lines matched
    if not prediction:
        for line in [l.strip() for l in cleaned.splitlines() if l.strip()]:
            if is_compound_prediction(line):
                logger.error("Remote model returned ambiguous standalone compound prediction.")
                raise RuntimeError(
                    "The remote model did not provide a recognizable prediction."
                )

        standalone_match = re.search(
            r"(?im)^\s*(?:[\d]+[\.\)]\s*|[-*]\s*)?(genuine|misleading|inconclusive|uncertain|unverified)\s*[.!]?\s*$",
            cleaned,
        )
        if standalone_match:
            prediction = normalize_prediction_label(standalone_match.group(1))

    # 4. Fallback: conservative check for prediction line with clear uncertainty phrasing
    if not prediction:
        uncertainty_line_match = re.search(
            r"(?im)^\s*(?:[\d]+[\.\)]\s*|[-*]\s*)?(?:prediction|verdict|classification|decision)\s*[:\-]\s*.*?\b(inconclusive|uncertain|unverified|cannot determine|unable to determine|insufficient (?:visual )?evidence)\b",
            cleaned,
        )
        if uncertainty_line_match:
            prediction = "Inconclusive"

    if not prediction:
        logger.error("Remote model did not provide a recognizable prediction.")
        raise RuntimeError(
            "The remote model did not provide a recognizable prediction."
        )

    reason_match = re.search(
        r"(?ims)^\s*(?:[\d]+[\.\)]\s*|[-*]\s*)?Reason\s*[:\-]\s*(.*?)(?=^\s*(?:[\d]+[\.\)]\s*|[-*]\s*)?Visual Evidence\s*[:\-]|\Z)",
        cleaned,
    )
    evidence_match = re.search(
        r"(?ims)^\s*(?:[\d]+[\.\)]\s*|[-*]\s*)?Visual Evidence\s*[:\-]\s*(.*)\Z",
        cleaned,
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match and reason_match.group(1).strip()
        else "No separate explanation was provided by the remote model."
    )
    visual_evidence = (
        evidence_match.group(1).strip()
        if evidence_match and evidence_match.group(1).strip()
        else "No separate visual evidence field was provided by the remote model."
    )

    if not reason or not visual_evidence:
        logger.error("The remote model returned an empty reason or visual evidence field.")
        raise RuntimeError(
            "The remote model returned an empty reason or visual evidence field."
        )

    # Safety correction: detect explicit claim-level uncertainty about whether the overall
    # image-caption claim can be verified. If a binary verdict conflicts with that uncertainty,
    # return Inconclusive. Incidental unknown details are preserved.
    if prediction in {"Genuine", "Misleading"} and has_claim_level_uncertainty(reason):
        # A Misleading prediction supported by explicit contradictory visual evidence
        # remains Misleading, rather than being downgraded merely because external
        # verification of the claimed event is impossible.
        if prediction == "Misleading" and (
            has_explicit_visual_contradiction(reason)
            or has_explicit_visual_contradiction(visual_evidence)
        ):
            logger.debug(
                "Misleading prediction supported by explicit contradictory visual evidence preserved despite general uncertainty phrasing."
            )
        else:
            logger.warning(
                "Binary verdict '%s' contradicted by claim-level uncertainty. Overriding prediction to 'Inconclusive'.",
                prediction,
            )
            prediction = "Inconclusive"

    model_used = payload.get("model") or primary_model

    return {
        "prediction": prediction,
        "reason": reason,
        "visual_evidence": visual_evidence,
        "model": model_used,
    }

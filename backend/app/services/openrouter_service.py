
import base64
import os
import re

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openrouter/free"


def analyze_remote_image(
    image_bytes: bytes,
    content_type: str,
    caption: str,
) -> dict:
    """Analyze an image-caption pair using OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    if not image_bytes:
        raise ValueError("Image data is empty.")

    if not caption.strip():
        raise ValueError("Caption cannot be empty.")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if content_type not in allowed_types:
        content_type = "image/jpeg"

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{content_type};base64,{encoded_image}"

    prompt = (
        "Analyze whether the image visually supports the caption. "
        "Do not invent visual details or claim that an unverified event is true. "
        "If the claim cannot be established from the image alone, state that limitation.\n\n"
        f"Caption: {caption}\n\n"
        "Return these exact fields on separate lines:\n"
        "Prediction: Genuine or Misleading\n"
        "Reason: A short explanation\n"
        "Visual Evidence: Details visible in the image"
    )

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "OOC-Verify",
            },
            json={
                "model": MODEL_NAME,
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
                "max_tokens": 350,
            },
            timeout=90,
        )
        response.raise_for_status()
    except requests.RequestException:
        raise RuntimeError(
            "Remote model request failed. Check OpenRouter availability, "
            "rate limits, and API configuration."
        ) from None

    try:
        payload = response.json()
        model_output = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError(
            "OpenRouter returned an unexpected response format."
        ) from None

    if not isinstance(model_output, str) or not model_output.strip():
        raise RuntimeError("The remote model returned an empty response.")

    cleaned = re.sub(r"[*_`#]", "", model_output).strip()

    prediction_match = re.search(
        r"(?im)^\s*Prediction\s*:\s*(Genuine|Misleading)\b",
        cleaned,
    )
    reason_match = re.search(
        r"(?ims)^\s*Reason\s*:\s*(.*?)(?=^\s*Visual Evidence\s*:|\Z)",
        cleaned,
    )
    evidence_match = re.search(
        r"(?ims)^\s*Visual Evidence\s*:\s*(.*)\Z",
        cleaned,
    )

    
    # If the model uses a different format, retry parsing common alternatives.
    if not prediction_match:
        prediction_match = re.search(
            r"(?i)\b(Genuine|Misleading)\b",
            cleaned,
        )

    if not reason_match:
        reason_match = re.search(
            r"(?is)\bReason\s*[:\-]\s*(.+?)(?=\bVisual Evidence\b|\Z)",
            cleaned,
        )

    if not evidence_match:
        evidence_match = re.search(
            r"(?is)\bVisual Evidence\s*[:\-]\s*(.+)\Z",
            cleaned,
        )

    if not prediction_match:
        raise RuntimeError(
            "The remote model did not provide a recognizable prediction."
        )

    prediction = prediction_match.group(1).capitalize()

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else "The remote model did not provide a separate explanation."
    )

    visual_evidence = (
        evidence_match.group(1).strip()
        if evidence_match
        else "The remote model did not provide a separate visual evidence field."
    )

    reason = reason_match.group(1).strip()
    visual_evidence = evidence_match.group(1).strip()

    if not reason or not visual_evidence:
        raise RuntimeError(
            "The remote model returned empty reason or visual evidence."
        )

    return {
        "prediction": prediction_match.group(1).capitalize(),
        "reason": reason,
        "visual_evidence": visual_evidence,
        "model": payload.get("model", MODEL_NAME),
    }
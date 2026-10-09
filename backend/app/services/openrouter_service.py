
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
    """Analyze whether an image visually supports its caption."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    if not image_bytes:
        raise ValueError("Image data is empty.")

    if not caption.strip():
        raise ValueError("Caption cannot be empty.")

    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        content_type = "image/jpeg"

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{content_type};base64,{encoded_image}"

    prompt = (
        "Analyze whether the image visually supports the caption. "
        "Judge visual consistency only; do not claim external events are verified. "
        "Return exactly these fields on separate lines:\n"
        "Prediction: Genuine or Misleading\n"
        "Reason: A short explanation\n"
        "Visual Evidence: Details visible in the image\n\n"
        f"Caption: {caption}"
    )

    try:
        print("DEBUG: Calling OpenRouter from openrouter_service.py", flush=True)
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
        print("DEBUG: OpenRouter HTTP status:", response.status_code, flush=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        if exc.response is not None:
            print(
                "OpenRouter error:",
                exc.response.status_code,
                exc.response.text[:1000],
            )
    else:
        print("OpenRouter connection error:", str(exc))

    raise RuntimeError(
        "Remote model request failed. Check the backend logs for details."
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

    # Prefer an explicitly labelled prediction.
    prediction_match = re.search(
        r"(?im)^\s*(?:prediction|verdict|classification|decision)\s*[:\-]\s*(genuine|misleading)\b",
        cleaned,
    )

    # Fallback: accept a clear standalone prediction line.
    if not prediction_match:
        prediction_match = re.search(
            r"(?im)^\s*(genuine|misleading)\s*[.!]?\s*$",
            cleaned,
        )

    if not prediction_match:
        raise RuntimeError(
            "The remote model did not provide a recognizable prediction. "
            f"Model response: {cleaned[:800]}"
        )

    prediction = prediction_match.group(1).capitalize()

    reason_match = re.search(
        r"(?ims)^\s*Reason\s*:\s*(.*?)(?=^\s*Visual Evidence\s*:|\Z)",
        cleaned,
    )
    evidence_match = re.search(
        r"(?ims)^\s*Visual Evidence\s*:\s*(.*)\Z",
        cleaned,
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else "No separate explanation was provided by the remote model."
    )
    visual_evidence = (
        evidence_match.group(1).strip()
        if evidence_match
        else "No separate visual evidence field was provided by the remote model."
    )

    if not reason or not visual_evidence:
        raise RuntimeError(
            "The remote model returned an empty reason or visual evidence field."
        )

    return {
        "prediction": prediction,
        "reason": reason,
        "visual_evidence": visual_evidence,
        "model": payload.get("model", MODEL_NAME),
    }
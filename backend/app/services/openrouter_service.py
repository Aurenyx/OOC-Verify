
import base64
import logging
import os
import re

from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/free")


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
        "Prediction: Genuine or Misleading\n"
        "Reason: A short explanation\n"
        "Visual Evidence: Details visible in the image\n\n"
        f"Caption: {caption}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "OOC-Verify",
    }
    payload_data = {
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
    }

    try:
        print("DEBUG: Calling OpenRouter from openrouter_service.py", flush=True)
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload_data,
            timeout=90,
        )
        print("DEBUG: OpenRouter HTTP status:", response.status_code, flush=True)
    except requests.Timeout as exc:
        print(f"OpenRouter connection timeout: {exc}", flush=True)
        logger.error(f"OpenRouter connection timeout: {exc}")
        raise RuntimeError("Remote model request timed out.") from None
    except requests.ConnectionError as exc:
        print(f"OpenRouter connection error: {exc}", flush=True)
        logger.error(f"OpenRouter connection error: {exc}")
        raise RuntimeError(f"Failed to connect to remote model service: {exc}") from None
    except requests.RequestException as exc:
        print(f"OpenRouter request error: {exc}", flush=True)
        logger.error(f"OpenRouter request error: {exc}")
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
        print(f"OpenRouter error: {response.status_code} {safe_detail}", flush=True)
        logger.error(f"OpenRouter error {response.status_code}: {safe_detail}")
        raise RuntimeError(
            f"Remote model request failed with status {response.status_code}: {safe_detail}"
        )

    try:
        payload = response.json()
    except Exception as exc:
        safe_snippet = response.text[:1000].replace(api_key, "[REDACTED]")
        print(f"OpenRouter invalid JSON: {exc}. Snippet: {safe_snippet}", flush=True)
        logger.error(f"OpenRouter invalid JSON: {exc}. Snippet: {safe_snippet}")
        raise RuntimeError(f"OpenRouter returned invalid JSON: {exc}") from None

    if not isinstance(payload, dict):
        print(f"OpenRouter unexpected payload type: {type(payload)}", flush=True)
        logger.error(f"OpenRouter unexpected payload type: {type(payload)}")
        raise RuntimeError("OpenRouter returned an unexpected response format: root is not a dictionary.")

    if "error" in payload:
        error_info = payload["error"]
        msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
        safe_msg = str(msg).replace(api_key, "[REDACTED]")
        print(f"OpenRouter API error in response payload: {safe_msg}", flush=True)
        logger.error(f"OpenRouter API error in response payload: {safe_msg}")
        raise RuntimeError(f"OpenRouter API error: {safe_msg}")

    try:
        choices = payload.get("choices")
        if not choices or not isinstance(choices, list):
            print(f"OpenRouter response missing choices: {payload}", flush=True)
            logger.error(f"OpenRouter response missing choices: {payload}")
            raise RuntimeError("OpenRouter returned no choices in response.")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            print(f"OpenRouter choice missing message object: {choices[0]}", flush=True)
            logger.error(f"OpenRouter choice missing message object: {choices[0]}")
            raise RuntimeError("OpenRouter returned an unexpected response format.")

        model_output = message.get("content")
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"OpenRouter parsing error: {exc}", flush=True)
        logger.error(f"OpenRouter parsing error: {exc}")
        raise RuntimeError("OpenRouter returned an unexpected response format.") from None

    if not isinstance(model_output, str) or not model_output.strip():
        print(f"The remote model returned an empty response. Choices: {choices}", flush=True)
        logger.error(f"The remote model returned an empty response: {choices}")
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
        print(f"OpenRouter unrecognized prediction. Full output:\n{model_output}", flush=True)
        logger.error(f"OpenRouter unrecognized prediction: {model_output}")
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
        if reason_match and reason_match.group(1).strip()
        else "No separate explanation was provided by the remote model."
    )
    visual_evidence = (
        evidence_match.group(1).strip()
        if evidence_match and evidence_match.group(1).strip()
        else "No separate visual evidence field was provided by the remote model."
    )

    if not reason or not visual_evidence:
        print(f"OpenRouter empty reason or visual evidence. Full output:\n{model_output}", flush=True)
        logger.error(f"OpenRouter empty reason or visual evidence: {model_output}")
        raise RuntimeError(
            "The remote model returned an empty reason or visual evidence field."
        )

    model_used = payload.get("model") or MODEL_NAME

    return {
        "prediction": prediction,
        "reason": reason,
        "visual_evidence": visual_evidence,
        "model": model_used,
    }
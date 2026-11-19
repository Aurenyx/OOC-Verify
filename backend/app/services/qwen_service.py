import gc
import logging
import math
import os
from pathlib import Path
import time
from typing import Optional, Tuple, Union

from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger("uvicorn.error")


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_MAX_IMAGE_EDGE = 1024
DEFAULT_MAX_IMAGE_PIXELS = 1003520  # 1280 * 28 * 28

_model = None
_processor = None


def _inspect_and_log_model_state(model: torch.nn.Module, init_elapsed: float) -> None:
    """Inspect and log actual tensor dtypes, module classes, and device map."""
    device_map = getattr(model, "hf_device_map", None)
    logger.info("[QWEN-INIT] Model initialization completed in %.3fs", init_elapsed)
    logger.info("[QWEN-DEVICE-MAP] Loaded hf_device_map: %s", device_map)

    try:
        # 1. Vision tower inspection (check actual Linear module, not Conv3d patch_embed)
        visual_mod = getattr(model, "visual", None) or getattr(
            getattr(model, "model", None), "visual", None
        )
        if visual_mod is not None:
            vis_linear = visual_mod.blocks[0].attn.qkv
            vis_type = type(vis_linear).__name__
            vis_dtype = getattr(vis_linear.weight, "dtype", "unknown")
            vis_device = getattr(vis_linear.weight, "device", "unknown")
            is_vis_unquantized = vis_type == "Linear" and str(vis_dtype) in (
                "torch.float16",
                "torch.bfloat16",
            )
            logger.info(
                "[QWEN-VERIFY] Vision block 0 linear (attn.qkv): class=%s, device=%s, dtype=%s (unquantized=%s)",
                vis_type,
                vis_device,
                vis_dtype,
                is_vis_unquantized,
            )
        else:
            logger.warning("[QWEN-VERIFY] Vision module could not be located on model.")

        # 2. Decoder layer inspection (check both first and last layers)
        layer0_proj = model.model.language_model.layers[0].mlp.gate_proj
        last_layer_proj = model.model.language_model.layers[-1].mlp.gate_proj
        logger.info(
            "[QWEN-VERIFY] Decoder layer 0 gate_proj: class=%s, device=%s, dtype=%s",
            type(layer0_proj).__name__,
            getattr(layer0_proj.weight, "device", "unknown"),
            getattr(layer0_proj.weight, "dtype", "unknown"),
        )
        logger.info(
            "[QWEN-VERIFY] Decoder layer 35 gate_proj: class=%s, device=%s, dtype=%s",
            type(last_layer_proj).__name__,
            getattr(last_layer_proj.weight, "device", "unknown"),
            getattr(last_layer_proj.weight, "dtype", "unknown"),
        )

        # 3. Tied embeddings and lm_head inspection (diagnostic only)
        embed_weight = model.model.language_model.embed_tokens.weight
        lm_head_weight = model.lm_head.weight
        is_tied = (lm_head_weight is embed_weight) or (
            hasattr(lm_head_weight, "data_ptr")
            and hasattr(embed_weight, "data_ptr")
            and lm_head_weight.data_ptr() == embed_weight.data_ptr()
        )
        logger.info(
            "[QWEN-VERIFY-DIAG] Embeddings: device=%s, dtype=%s | lm_head: device=%s, dtype=%s (weights_tied=%s)",
            getattr(embed_weight, "device", "unknown"),
            getattr(embed_weight, "dtype", "unknown"),
            getattr(lm_head_weight, "device", "unknown"),
            getattr(lm_head_weight, "dtype", "unknown"),
            is_tied,
        )

        # 4. Resolve exact decoder layer placement accounting for parent module mappings
        num_layers = len(model.model.language_model.layers)
        cuda_layers, cpu_layers, disk_layers, unknown_layers = [], [], [], []

        for i in range(num_layers):
            layer_name = f"model.language_model.layers.{i}"
            resolved_dev = None

            if isinstance(device_map, dict):
                if layer_name in device_map:
                    resolved_dev = device_map[layer_name]
                else:
                    parts = layer_name.split(".")
                    for p_len in range(len(parts) - 1, 0, -1):
                        prefix = ".".join(parts[:p_len])
                        if prefix in device_map:
                            resolved_dev = device_map[prefix]
                            break
                    if resolved_dev is None and "" in device_map:
                        resolved_dev = device_map[""]

            # Fall back to parameter device if device_map did not resolve
            if resolved_dev is None:
                param_dev = getattr(
                    model.model.language_model.layers[i].mlp.gate_proj.weight,
                    "device",
                    None,
                )
                if param_dev is not None:
                    resolved_dev = str(param_dev)

            # Categorize device
            dev_str = (
                str(resolved_dev).lower()
                if resolved_dev is not None
                else "unknown"
            )
            if "cuda" in dev_str or dev_str == "0":
                cuda_layers.append(i)
            elif "cpu" in dev_str:
                cpu_layers.append(i)
            elif "disk" in dev_str:
                disk_layers.append(i)
            else:
                unknown_layers.append(i)

        logger.info(
            "[QWEN-VERIFY] Decoder layer distribution (%d total): %d on CUDA, %d on CPU, %d on disk, %d unknown",
            num_layers,
            len(cuda_layers),
            len(cpu_layers),
            len(disk_layers),
            len(unknown_layers),
        )
        if cpu_layers or disk_layers:
            logger.warning(
                "[QWEN-VERIFY] Layers offloaded: CPU=%s, Disk=%s",
                cpu_layers,
                disk_layers,
            )
        if unknown_layers:
            logger.warning(
                "[QWEN-VERIFY] Layers with undetermined placement: %s",
                unknown_layers,
            )

    except Exception as inspect_err:
        logger.warning(
            "[QWEN-VERIFY] Diagnostic inspection encountered an error: %s",
            inspect_err,
        )


def load_model():
    global _model, _processor

    if _model is not None:
        return

    # Attempt to load backend/.env if available
    backend_env = Path(__file__).resolve().parents[2] / ".env"
    if backend_env.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=backend_env)
        except Exception:
            pass

    quant_mode = os.getenv("QWEN_QUANTIZATION", "4bit").strip().lower()
    logger.info(
        "[QWEN-INIT] Starting Qwen2.5-VL initialization in mode='%s'...",
        quant_mode,
    )
    start_init = time.perf_counter()

    if quant_mode == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401 - verify package availability

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=["visual", "model.visual", "lm_head"],
            )
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "4.0GiB", "cpu": "4.5GiB"},
            )
            logger.info("[QWEN-INIT] 4-bit NF4 model loaded successfully.")
        except Exception as exc:
            _model = None
            _processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.critical(
                "[QWEN-INIT-FAIL] 4-bit load failed: %s. "
                "No in-process fallback will be attempted to avoid CUDA memory corruption. "
                "To rollback: restart process with $env:QWEN_QUANTIZATION='fp16'.",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"4-bit Qwen2.5-VL load failed: {exc}. Restart with $env:QWEN_QUANTIZATION='fp16' to use FP16 baseline."
            ) from exc

    elif quant_mode == "fp16":
        logger.info(
            "[QWEN-INIT] Loading baseline FP16 model with CPU offload..."
        )
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={
                0: "4.5GiB",
                "cpu": "12GiB",
            },
            offload_folder="qwen_offload",
        )
        logger.info("[QWEN-INIT] Baseline FP16 model loaded successfully.")

    else:
        raise ValueError(
            f"Invalid QWEN_QUANTIZATION='{quant_mode}'. Supported values: '4bit', 'fp16'."
        )

    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    init_elapsed = time.perf_counter() - start_init
    _inspect_and_log_model_state(_model, init_elapsed)
    print("Qwen2.5-VL loaded successfully.")


def prepare_image_for_qwen(
    image_input: Union[str, Path, Image.Image],
    max_edge: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    Validate, normalize, and downscale an image for Qwen vision processing.

    Guarantees:
    1. Aspect ratio is strictly preserved.
    2. Small images below limits are NEVER upscaled (scale capped at 1.0).
    3. Converted to RGB mode (safely handling RGBA, L, CMYK, P).
    4. Limits configurable via environment variables or explicit parameters:
       - QWEN_MAX_IMAGE_EDGE (default 1024)
       - QWEN_MAX_IMAGE_PIXELS (default 1003520 = 1280 * 28 * 28)
    5. Returns (prepared_image, original_dimensions, resized_dimensions).
    """
    if max_edge is None:
        try:
            max_edge = int(os.getenv("QWEN_MAX_IMAGE_EDGE", str(DEFAULT_MAX_IMAGE_EDGE)))
        except (ValueError, TypeError):
            max_edge = DEFAULT_MAX_IMAGE_EDGE

    if max_pixels is None:
        try:
            max_pixels = int(os.getenv("QWEN_MAX_IMAGE_PIXELS", str(DEFAULT_MAX_IMAGE_PIXELS)))
        except (ValueError, TypeError):
            max_pixels = DEFAULT_MAX_IMAGE_PIXELS

    if isinstance(image_input, (str, Path)):
        file_path = Path(image_input)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_input}")
        if file_path.stat().st_size == 0:
            raise ValueError(f"Image file is empty (0 bytes): {image_input}")
        try:
            img = Image.open(file_path)
            img.load()
        except Exception as exc:
            if isinstance(exc, (ValueError, FileNotFoundError)):
                raise
            raise ValueError(f"Corrupted or invalid image file '{image_input}': {exc}") from exc
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_input).__name__}. Expected str, Path, or PIL.Image.Image."
        )

    if img.mode != "RGB":
        img = img.convert("RGB")

    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

    scale = 1.0
    if max_edge > 0 and max(orig_w, orig_h) > max_edge:
        scale = min(scale, max_edge / max(orig_w, orig_h))
    if max_pixels > 0 and (orig_w * orig_h) > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (orig_w * orig_h)))

    # NEVER upscale: only downscale if scale < 1.0
    if scale < 1.0:
        target_w = max(1, int(round(orig_w * scale)))
        target_h = max(1, int(round(orig_h * scale)))
        # Strictly guarantee dimensions do not exceed limits due to integer rounding
        if max_edge > 0 and (target_w > max_edge or target_h > max_edge):
            edge_scale = max_edge / max(target_w, target_h)
            target_w = max(1, int(math.floor(target_w * edge_scale)))
            target_h = max(1, int(math.floor(target_h * edge_scale)))
        if max_pixels > 0 and (target_w * target_h) > max_pixels:
            px_scale = math.sqrt(max_pixels / (target_w * target_h))
            target_w = max(1, int(math.floor(target_w * px_scale)))
            target_h = max(1, int(math.floor(target_h * px_scale)))
        resized_img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        resized_img = img

    return resized_img, (orig_w, orig_h), resized_img.size


def analyze_image(image_path: str, caption: str) -> str:
    start_total = time.perf_counter()
    load_model()

    start_prep = time.perf_counter()

    # Preprocess image input specifically for Qwen: resize if exceeding limits, preserve aspect ratio, never upscale
    qwen_image, orig_dims, resized_dims = prepare_image_for_qwen(image_path)
    try:
        max_edge_val = int(os.getenv("QWEN_MAX_IMAGE_EDGE", str(DEFAULT_MAX_IMAGE_EDGE)))
    except (ValueError, TypeError):
        max_edge_val = DEFAULT_MAX_IMAGE_EDGE

    try:
        max_pixels_val = int(os.getenv("QWEN_MAX_IMAGE_PIXELS", str(DEFAULT_MAX_IMAGE_PIXELS)))
    except (ValueError, TypeError):
        max_pixels_val = DEFAULT_MAX_IMAGE_PIXELS

    logger.info(
        "[QWEN-PREP] Original image dimensions: %dx%d | Qwen input dimensions: %dx%d (max_edge=%d, max_pixels=%d)",
        orig_dims[0],
        orig_dims[1],
        resized_dims[0],
        resized_dims[1],
        max_edge_val,
        max_pixels_val,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": qwen_image,
                    "max_pixels": max_pixels_val,
                },
                {
                    "type": "text",
                    "text": f"""
You are a careful multimodal image-caption verification assistant for an Out-of-Context (OOC) misinformation detection system.

Your task is to analyze the supplied image and caption independently and explain exactly what the image supports, contradicts, or cannot establish.

CAPTION TO VERIFY:
{caption}

Follow these steps:

OBSERVE THE IMAGE
Identify the main objects, people, actions, environment, and other relevant visible details. Describe only what is actually visible. Do not invent identities, locations, dates, events, intentions, or causes.

EXTRACT THE CAPTION'S CLAIMS
Separate directly observable claims (objects, clothing, actions, visible conditions) from contextual claims (specific event, person identity, organization, location, date, historical incident, or reported occurrence).

COMPARE EACH CLAIM WITH THE IMAGE
For each important claim, determine:

SUPPORTED: The visible image provides clear evidence consistent with the claim.

CONTRADICTED: The visible image clearly conflicts with the claim.

UNVERIFIED: The image alone cannot establish or refute the claim.

Do not treat missing visual evidence as proof that a claim is false. A photograph may be visually compatible with an event without proving that it depicts that specific event.

MAKE A CAREFUL ASSESSMENT

Genuine: The relevant claims are visually consistent, and there is no material contradiction. Do not claim that contextual facts are independently verified when they are not.

Misleading: There is a clear, material contradiction between the caption and the image.

Inconclusive: The key claim depends on context that the image cannot establish, and there is insufficient evidence to determine whether it is true or false.

For mixed captions, assess the individual claims and explain any material disagreement.

WRITE A SPECIFIC EXPLANATION
Cite concrete visible details. Explain which part of the caption matches or conflicts with those details. State explicitly when a location, date, identity, event, or historical connection cannot be determined from pixels alone.

Do not use outside knowledge to invent evidence. Do not confuse topic similarity with proof of an event. Do not claim that something is absent unless it is relevant and reasonably observable.

Return exactly these five lines:
Prediction: Genuine, Misleading, or Inconclusive
Reason: A specific explanation connecting the caption's claims to visible evidence
Visual Evidence: Concrete objects, actions, and scene details actually visible
Claim Assessment: Important claims with their SUPPORTED, CONTRADICTED, or UNVERIFIED status
Caption Type: DIRECT_VISUAL, CONTEXTUAL, or MIXED
""",
                },
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",)

    # Place token-level inputs on embedding device.
    # Ensure vision tensors match the vision tower device (preventing CPU/CUDA mismatch).
    input_device = _model.model.language_model.embed_tokens.weight.device
    visual_mod = getattr(_model, "visual", None) or getattr(
        getattr(_model, "model", None), "visual", None
    )
    vis_device = (
        visual_mod.patch_embed.proj.weight.device
        if visual_mod is not None and hasattr(visual_mod, "patch_embed")
        else input_device
    )

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if key in ("pixel_values", "image_grid_thw"):
                inputs[key] = value.to(vis_device)
            else:
                inputs[key] = value.to(input_device)
            
    elapsed_prep = time.perf_counter() - start_prep

    # Extract visual token count and grid if available
    visual_tokens = None
    grid_thw = None
    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
        grid_list = inputs.image_grid_thw.tolist()
        if grid_list and len(grid_list[0]) == 3:
            grid_thw = grid_list[0]
            # In Qwen2.5-VL with spatial merge size 2, visual tokens = (h // 2) * (w // 2)
            visual_tokens = (grid_thw[1] // 2) * (grid_thw[2] // 2)

    total_tokens = (
        inputs.input_ids.shape[1]
        if hasattr(inputs, "input_ids") and inputs.input_ids is not None
        else None
    )

    logger.info(
        "[QWEN-PREP] Preprocessing completed in %.3fs | Grid THW: %s | Visual tokens: %s | Total input tokens: %s",
        elapsed_prep,
        grid_thw,
        visual_tokens if visual_tokens is not None else "unavailable",
        total_tokens if total_tokens is not None else "unavailable",
    )

    start_gen = time.perf_counter()
    with torch.inference_mode():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=150,
        )
    elapsed_gen = time.perf_counter() - start_gen

    start_decode = time.perf_counter()
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = _processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    elapsed_decode = time.perf_counter() - start_decode
    elapsed_total = time.perf_counter() - start_total

    raw_response = output[0] if output else ""
    raw_preview = (
        raw_response[:2000] + "... [truncated]"
        if len(raw_response) > 2000
        else raw_response
    )
    final_result = raw_response.strip()

    logger.info(
        "[QWEN-TIMING] Timing breakdown: preprocessing=%.3fs, generation=%.3fs, decoding=%.3fs, total=%.3fs",
        elapsed_prep,
        elapsed_gen,
        elapsed_decode,
        elapsed_total,
    )
    logger.info(
        "[QWEN-RAW-RESPONSE] Raw decoded output (%d chars):\n%s",
        len(raw_response),
        raw_preview,
    )
    logger.info(
        "[QWEN-FINAL-RESULT] Final parsed result returned by analyze_image():\n%s",
        final_result,
    )

    return final_result
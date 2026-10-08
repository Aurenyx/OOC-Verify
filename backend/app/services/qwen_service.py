import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

_model = None
_processor = None


def load_model():
    global _model, _processor

    if _model is not None:
        return

    print("Loading Qwen2.5-VL model...")

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

    _processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Qwen2.5-VL loaded successfully.")


def analyze_image(image_path: str, caption: str) -> str:
    load_model()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": f"""
You verify whether an image supports the given caption.

CAPTION TO VERIFY:
{caption}

First classify the caption as either:

DIRECT_VISUAL:
The caption describes something directly visible in the image, such as a person, object, clothing, action, or visible scene.

CONTEXTUAL:
The caption makes a claim about an event, organization, person, location, date, quantity, announcement, or other information that cannot be established from visual appearance alone.

For DIRECT_VISUAL captions:
Check only whether the described content is visibly present.
If it is visible, predict Genuine.
If it is not visible, predict Misleading.
Do not require external evidence or event context.

For CONTEXTUAL captions:
Check whether the image itself provides meaningful visual evidence for the specific claim.
A generic image related to the topic is not sufficient.
If the image does not establish the specific claim or context, predict Misleading.

IMPORTANT:
Do not confuse topic similarity with visual evidence.
Do not invent information that is not visible.
Do not use outside knowledge to establish whether an event actually happened.

Return exactly three lines:

Prediction: Genuine or Misleading
Reason: <short reason>
Visual Evidence: <what is actually visible>
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
    return_tensors="pt",
)


    with torch.inference_mode():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=150,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = _processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output[0].strip()
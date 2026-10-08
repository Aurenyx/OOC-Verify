import io
import logging
from typing import Optional, Union

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import settings

logger = logging.getLogger(__name__)


class CLIPService:
    """
    CLIP service for image-caption semantic alignment.

    Returns cosine similarity between normalized CLIP image and text
    embeddings. This matches the feature used to train the
    NewsCLIPpings alignment classifier.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.CLIP_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self._is_loaded = False

        self.device_name = (
            torch.cuda.get_device_name(0)
            if self.device == "cuda"
            else "CPU"
        )

    def load_model(self) -> None:
        """Load CLIP model once."""

        if (
            self._is_loaded
            and self.model is not None
            and self.processor is not None
        ):
            return

        logger.info(
            f"Loading CLIP model '{self.model_name}' "
            f"on {self.device} ({self.device_name})..."
        )

        self.model = CLIPModel.from_pretrained(
            self.model_name
        )

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True

        logger.info(
            f"CLIP model '{self.model_name}' loaded successfully."
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def compute_alignment_score(
        self,
        image_input: Union[bytes, Image.Image],
        caption: str,
    ) -> float:
        """
        Compute cosine similarity between CLIP image and text embeddings.

        This matches the feature used by the NewsCLIPpings alignment
        classifier.
        """

        if (
            not self._is_loaded
            or self.model is None
            or self.processor is None
        ):
            self.load_model()

        # -----------------------------
        # Convert image
        # -----------------------------

        if isinstance(image_input, bytes):
            image = Image.open(
                io.BytesIO(image_input)
            ).convert("RGB")

        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")

        else:
            raise ValueError(
                f"Unsupported image input type: {type(image_input)}"
            )

        cleaned_caption = caption.strip()

        if not cleaned_caption:
            raise ValueError(
                "Caption cannot be empty."
            )

        # -----------------------------
        # Prepare inputs
        # -----------------------------

        inputs = self.processor(
    text=[caption],
    images=image,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=77,
)

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # -----------------------------
        # Generate embeddings
        # -----------------------------

        with torch.inference_mode():

            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            image_embedding = (
                vision_outputs.pooler_output
            )

            image_embedding = (
                self.model.visual_projection(
                    image_embedding
                )
            )

            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            text_embedding = (
                text_outputs.pooler_output
            )

            text_embedding = (
                self.model.text_projection(
                    text_embedding
                )
            )

            # -----------------------------
            # Normalize embeddings
            # -----------------------------

            image_embedding = (
                image_embedding
                / image_embedding.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            text_embedding = (
                text_embedding
                / text_embedding.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            # -----------------------------
            # Cosine similarity
            # -----------------------------

            similarity = (
                image_embedding * text_embedding
            ).sum(dim=-1)

        return float(similarity.item())


# Singleton instance shared by the application
clip_service = CLIPService()


def get_clip_service() -> CLIPService:
    """Return the shared CLIP service instance."""
    return clip_service
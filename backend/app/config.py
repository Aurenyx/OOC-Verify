import os
from typing import List, Set
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "OOC-Verify Research Backend"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    # CORS configuration
    # Accepts comma-separated list of origins or FRONTEND_ORIGIN
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            f"{os.getenv('FRONTEND_ORIGIN', 'http://localhost:5173')},http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000",
        ).split(",")
        if origin.strip()
    ]

    # Image upload constraints
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_IMAGE_TYPES: Set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }
    ALLOWED_EXTENSIONS: Set[str] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    # Configuration modes
    SUPPORTED_CONFIGURATIONS: Set[str] = {
        "alignment_only",
        "mllm_only",
        "proposed",
    }

    # Multimodal CLIP model configuration
    CLIP_MODEL_NAME: str = os.getenv(
        "CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"
    )


settings = Settings()

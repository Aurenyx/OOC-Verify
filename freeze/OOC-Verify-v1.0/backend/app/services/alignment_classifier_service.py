from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "news_clippings"
    / "experiments"
    / "alignment_classifier.joblib"
)

_classifier = None


def load_model():
    global _classifier

    if _classifier is not None:
        return

    print("Loading trained alignment classifier...")

    _classifier = joblib.load(MODEL_PATH)

    print("Alignment classifier loaded successfully.")


def predict(alignment_score: float):
    load_model()

    features = np.asarray(
        [[alignment_score]],
        dtype=np.float32,
    )

    prediction = int(
        _classifier.predict(features)[0]
    )

    probabilities = _classifier.predict_proba(features)[0]

    confidence = float(
        probabilities[prediction]
    )

    if prediction == 1:
        label = "Misleading"
    else:
        label = "Genuine"

    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": {
            "genuine": float(probabilities[0]),
            "misleading": float(probabilities[1]),
        },
    }
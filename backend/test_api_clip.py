from PIL import ImagePath
import os
from starlette.testclient import TestClient
from app.main import app

def test_clip_integration():
    print("Testing FastAPI app lifespan and CLIP integration...")

    with TestClient(app) as client:
        # 1. Test /health
        health_resp = client.get("/api/v1/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        health_data = health_resp.json()
        print("Health check response:", health_data)
        assert health_data["status"] == "ok"
        assert health_data["clip"]["loaded"] is True
        print(f"CLIP Model: {health_data['clip']['model']} on {health_data['clip']['device']} ({health_data['clip']['device_name']})")

        # 2. Test /verify with image matching 'A person standing in front of a building'
        from pathlib import Path

        image_path = str(Path(__file__).parent / "test_images" / "test_image.jpg")
        assert os.path.exists(image_path), f"Missing {image_path}"

        caption1 = "A person standing in front of a building"
        with open(image_path, "rb") as f:
            resp1 = client.post(
                "/api/v1/verify",
                files={"image": ("test_image.jpg", f, "image/jpeg")},
                data={"caption": caption1, "dataset": "Custom Test"},
            )

        assert resp1.status_code == 200, f"Verify 1 failed: {resp1.text}"
        data1 = resp1.json()
        print("\n--- Response 1 ---")
        print("Caption:", caption1)
        print("CLIP Score (clip_score):", data1.get("clip_score"))
        print("Confidence Score:", data1.get("confidence_score"))
        print("Prediction:", data1.get("prediction"))
        print("Reason:", data1.get("reason"))
        print("Explanation:", data1.get("explanation"))
        print("Stages:", [(s["stage"], s["status"]) for s in data1.get("pipeline_stages", [])])

        assert data1.get("clip_score") is not None
        
        # In test_clip.py, this was ~20.6170
        assert -1.0 <= data1["clip_score"] <= 1.0
        
        assert data1["is_stub"] is False

        # 3. Test /verify with another caption: 'A dog running through a forest'
        caption2 = "A dog running through a forest"
        with open(image_path, "rb") as f:
            resp2 = client.post(
                "/api/v1/verify",
                files={"image": ("test_image.jpg", f, "image/jpeg")},
                data={"caption": caption2},
            )

        assert resp2.status_code == 200, f"Verify 2 failed: {resp2.text}"
        data2 = resp2.json()
        print("\n--- Response 2 ---")
        print("Caption:", caption2)
        print("CLIP Score (clip_score):", data2.get("clip_score"))
        # In test_clip.py, this was ~11.6205
        assert -1.0 <= data2["clip_score"] <= 1.0

        # Check that score for building caption is higher than forest caption
        assert data1["clip_score"] > data2["clip_score"], "Building caption should score higher than dog in forest"

        print("\nAll integration verification tests passed successfully!")

if __name__ == "__main__":
    test_clip_integration()

import os
import unittest
from unittest.mock import MagicMock, patch
import requests
from starlette.testclient import TestClient

from app.main import app
from app.services.openrouter_service import analyze_remote_image


class TestOpenRouterRemoteVerification(unittest.TestCase):
    def setUp(self):
        self.dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
        self.sample_caption = "A cat sitting on a rug."

    def test_missing_api_key_raises_runtime_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("OPENROUTER_API_KEY is not configured", str(ctx.exception))

    def test_empty_image_raises_value_error(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            with self.assertRaises(ValueError) as ctx:
                analyze_remote_image(b"", "image/jpeg", self.sample_caption)
            self.assertIn("Image data is empty", str(ctx.exception))

    def test_empty_caption_raises_value_error(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            with self.assertRaises(ValueError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", "   ")
            self.assertIn("Caption cannot be empty", str(ctx.exception))

    @patch("requests.post")
    def test_successful_analysis_genuine(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine\n"
                            "Reason: The image clearly depicts a tabby cat sitting on a woven rug.\n"
                            "Visual Evidence: A domestic cat is resting directly on a patterned floor rug."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["prediction"], "Genuine")
        self.assertEqual(
            result["reason"],
            "The image clearly depicts a tabby cat sitting on a woven rug.",
        )
        self.assertEqual(
            result["visual_evidence"],
            "A domestic cat is resting directly on a patterned floor rug.",
        )
        self.assertEqual(result["model"], "openrouter/free")

    @patch("requests.post")
    def test_successful_analysis_misleading(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Misleading\n"
                            "Reason: The image contains a golden retriever dog, not a cat.\n"
                            "Visual Evidence: A canine with golden fur is lying on grass outdoors."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["prediction"], "Misleading")
        self.assertIn("golden retriever", result["reason"])
        self.assertEqual(result["model"], "meta-llama/llama-3.2-11b-vision-instruct:free")

    @patch("requests.post")
    def test_http_error_does_not_expose_api_key(self, mock_post):
        secret_key = "secret_sk_live_99998888"
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = f"Unauthorized: invalid key {secret_key}"
        mock_response.json.return_value = {"error": {"message": f"Invalid API key {secret_key}"}}
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": secret_key}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

            error_text = str(ctx.exception)
            self.assertIn("401", error_text)
            self.assertNotIn(secret_key, error_text)
            self.assertIn("[REDACTED]", error_text)

    @patch("requests.post")
    def test_network_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout("Connection timed out after 90s")

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("timed out", str(ctx.exception).lower())

    @patch("requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Failed to resolve host")

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("failed to connect", str(ctx.exception).lower())

    @patch("requests.post")
    def test_invalid_json_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON token")
        mock_response.text = "<html>502 Bad Gateway from Cloudflare</html>"
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("invalid json", str(ctx.exception).lower())

    @patch("requests.post")
    def test_empty_model_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "   "}}]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("empty response", str(ctx.exception).lower())

    @patch("requests.post")
    def test_unrecognized_prediction(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I am an AI assistant and I cannot determine if this is true or false."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("did not provide a recognizable prediction", str(ctx.exception))


class TestVerificationApiRemoteRoute(unittest.TestCase):
    def setUp(self):
        self.dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"

    def test_api_verify_remote_mode_success(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                return_value={
                    "prediction": "Misleading",
                    "reason": "The caption claims the person is wearing a white shirt, but it is black.",
                    "visual_evidence": "The person is wearing a black collared shirt.",
                    "model": "openrouter/free",
                },
            ):
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/verify",
                        files={"image": ("test.jpg", self.dummy_image, "image/jpeg")},
                        data={"caption": "A person wearing a white shirt"},
                    )

                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["prediction"], "Misleading")
                self.assertEqual(data["reason"], "The caption claims the person is wearing a white shirt, but it is black.")
                self.assertFalse(data["is_stub"])
                self.assertEqual(len(data["pipeline_stages"]), 4)

    def test_api_verify_remote_mode_upstream_502(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                side_effect=RuntimeError("Remote model request failed with status 502: Provider unavailable"),
            ):
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/verify",
                        files={"image": ("test.jpg", self.dummy_image, "image/jpeg")},
                        data={"caption": "A person wearing a white shirt"},
                    )

                self.assertEqual(resp.status_code, 502)
                data = resp.json()
                self.assertIn("Remote verification failed", data["detail"])
                self.assertIn("502", data["detail"])

    def test_api_verify_remote_mode_missing_api_key_500(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": ""}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                side_effect=RuntimeError("OPENROUTER_API_KEY is not configured."),
            ):
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/verify",
                        files={"image": ("test.jpg", self.dummy_image, "image/jpeg")},
                        data={"caption": "A person wearing a white shirt"},
                    )

                self.assertEqual(resp.status_code, 500)
                data = resp.json()
                self.assertIn("missing OPENROUTER_API_KEY", data["detail"])

    def test_api_verify_empty_caption_400(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote"}):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/verify",
                    files={"image": ("test.jpg", self.dummy_image, "image/jpeg")},
                    data={"caption": "   "},
                )

            self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()

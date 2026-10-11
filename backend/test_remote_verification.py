import io
import os
import socket
import sys
import unittest
from unittest.mock import MagicMock, patch
import requests
from starlette.testclient import TestClient

from app.main import app
from app.services.openrouter_service import (
    DEFAULT_VISION_MODELS,
    analyze_remote_image,
    get_candidate_models,
)

_orig_socket_connect = socket.socket.connect


def setUpModule():
    def guard_connect(self_sock, addr):
        # Allow loopback/local connections (required by asyncio loopback and TestClient on Windows)
        host = addr[0] if isinstance(addr, tuple) and addr else addr
        if host in ("127.0.0.1", "localhost", "::1"):
            return _orig_socket_connect(self_sock, addr)
        raise AssertionError(f"Real external network connection to {addr} attempted during test execution!")

    socket.socket.connect = guard_connect


def tearDownModule():
    socket.socket.connect = _orig_socket_connect


class TestOpenRouterRemoteVerification(unittest.TestCase):
    def setUp(self):
        self.dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
        self.sample_caption = "A cat sitting on a rug."
        self._stdout_trap = io.StringIO()
        self._orig_stdout = sys.stdout
        sys.stdout = self._stdout_trap

    def tearDown(self):
        sys.stdout = self._orig_stdout

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

    @patch("app.services.openrouter_service.requests.post")
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

    @patch("app.services.openrouter_service.requests.post")
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

    @patch("app.services.openrouter_service.requests.post")
    def test_successful_analysis_explicit_inconclusive(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Inconclusive\n"
                            "Reason: The image is too blurry to confirm or refute the caption.\n"
                            "Visual Evidence: Low-resolution unidentifiable objects on a street."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["prediction"], "Inconclusive")
        self.assertIn("too blurry", result["reason"])
        self.assertEqual(result["model"], "openrouter/free")

    @patch("app.services.openrouter_service.requests.post")
    def test_successful_analysis_normalized_uncertain(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Uncertain\n"
                            "Reason: The specific person mentioned in the caption cannot be clearly identified.\n"
                            "Visual Evidence: A distant crowd of people facing away from the camera."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["prediction"], "Inconclusive")
        self.assertIn("cannot be clearly identified", result["reason"])

    @patch("app.services.openrouter_service.requests.post")
    def test_successful_analysis_normalized_unverified(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Unverified\n"
                            "Reason: Visual contents neither confirm nor contradict the contextual event.\n"
                            "Visual Evidence: Generic conference room with anonymous attendees."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["prediction"], "Inconclusive")

    @patch("app.services.openrouter_service.requests.post")
    def test_successful_analysis_clear_uncertainty_wording(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Insufficient visual evidence\n"
                            "Reason: Visual evidence is insufficient to verify the claim.\n"
                            "Visual Evidence: Face is turned away and obscuring shadows."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["prediction"], "Inconclusive")

    @patch("app.services.openrouter_service.requests.post")
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

    @patch("app.services.openrouter_service.requests.post")
    def test_network_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout("Connection timed out after 90s")

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("timed out", str(ctx.exception).lower())

    @patch("app.services.openrouter_service.requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Failed to resolve host")

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)
            self.assertIn("failed to connect", str(ctx.exception).lower())

    @patch("app.services.openrouter_service.requests.post")
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

    @patch("app.services.openrouter_service.requests.post")
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

    @patch("app.services.openrouter_service.requests.post")
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

    @patch("app.services.openrouter_service.requests.post")
    def test_default_model_routing_uses_verified_vision_models(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "google/gemma-4-31b-it:free",
            "choices": [
                {
                    "message": {
                        "content": "Prediction: Genuine\nReason: Cat is visible.\nVisual Evidence: Tabby cat."
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}, clear=True):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["model"], "google/gemma-4-31b-it:free")
        called_json = mock_post.call_args[1]["json"]
        self.assertIn("models", called_json)
        self.assertLessEqual(len(called_json["models"]), 3)
        self.assertEqual(called_json["model"], "google/gemma-4-31b-it:free")
        self.assertEqual(
            called_json["models"],
            [
                "google/gemma-4-31b-it:free",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            ],
        )
        self.assertNotIn("dots-studio/dots-3-note-preview:free", called_json["models"])
        self.assertNotIn("openrouter/free", called_json["models"])

    @patch("app.services.openrouter_service.requests.post")
    def test_custom_model_routing_and_deduplication(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "choices": [
                {
                    "message": {
                        "content": "Prediction: Genuine\nReason: Cat is visible.\nVisual Evidence: Tabby cat."
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-12345",
                "OPENROUTER_MODEL": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            },
            clear=True,
        ):
            analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        called_json = mock_post.call_args[1]["json"]
        models_list = called_json["models"]
        self.assertEqual(models_list[0], "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
        self.assertEqual(models_list[1], "google/gemma-4-31b-it:free")
        self.assertEqual(len(models_list), len(set(models_list)), "Models list must not contain duplicates")
        self.assertLessEqual(len(models_list), 3)
        self.assertNotIn("dots-studio/dots-3-note-preview:free", models_list)

    @patch("app.services.openrouter_service.requests.post")
    def test_openrouter_free_env_overridden_with_verified_models(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "google/gemma-4-31b-it:free",
            "choices": [
                {
                    "message": {
                        "content": "Prediction: Genuine\nReason: Cat is visible.\nVisual Evidence: Tabby cat."
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-12345",
                "OPENROUTER_MODEL": "openrouter/free",
            },
            clear=True,
        ):
            analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        called_json = mock_post.call_args[1]["json"]
        self.assertNotIn("openrouter/free", called_json["models"])
        self.assertEqual(
            called_json["models"],
            [
                "google/gemma-4-31b-it:free",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            ],
        )
        self.assertNotIn("dots-studio/dots-3-note-preview:free", called_json["models"])

    def test_get_candidate_models_default_ordering(self):
        with patch.dict(os.environ, {}, clear=True):
            candidates = get_candidate_models()
        self.assertEqual(
            candidates,
            [
                "google/gemma-4-31b-it:free",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            ],
        )
        self.assertEqual(candidates, DEFAULT_VISION_MODELS)
        self.assertNotIn("dots-studio/dots-3-note-preview:free", candidates)

    def test_get_candidate_models_custom_env_override_and_capping(self):
        with patch.dict(
            os.environ,
            {"OPENROUTER_MODEL": "meta-llama/llama-3.2-11b-vision-instruct:free"},
            clear=True,
        ):
            candidates = get_candidate_models()
        self.assertEqual(
            candidates,
            [
                "meta-llama/llama-3.2-11b-vision-instruct:free",
                "google/gemma-4-31b-it:free",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            ],
        )
        self.assertLessEqual(len(candidates), 3)

    def test_get_candidate_models_deduplication(self):
        with patch.dict(
            os.environ,
            {"OPENROUTER_MODEL": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"},
            clear=True,
        ):
            candidates = get_candidate_models()
        self.assertEqual(
            candidates,
            [
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "google/gemma-4-31b-it:free",
            ],
        )
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_get_candidate_models_ignores_generic_openrouter_free_placeholder(self):
        for val in ["openrouter/free", "OPENROUTER/FREE", "  openrouter/free  "]:
            with patch.dict(os.environ, {"OPENROUTER_MODEL": val}, clear=True):
                candidates = get_candidate_models()
            self.assertEqual(
                candidates,
                [
                    "google/gemma-4-31b-it:free",
                    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                ],
            )
            self.assertNotIn("openrouter/free", candidates)

    @patch("app.services.openrouter_service.requests.post")
    def test_rejects_user_safety_safe_classification(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "nvidia/nemotron-3.5-content-safety:free",
            "choices": [
                {
                    "message": {
                        "content": "User Safety: safe"
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

            error_msg = str(ctx.exception)
            self.assertIn("content safety classification", error_msg)
            self.assertNotIn("Genuine", error_msg)

    @patch("app.services.openrouter_service.requests.post")
    def test_contradictory_verdict_genuine_with_claim_uncertainty_downgrades_to_inconclusive(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine\n"
                            "Reason: Visual evidence is insufficient to verify the claim from the image alone.\n"
                            "Visual Evidence: Generic conference room with anonymous figures."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Inconclusive")
        self.assertIn("insufficient to verify", result["reason"])

    @patch("app.services.openrouter_service.requests.post")
    def test_contradictory_verdict_misleading_with_claim_uncertainty_downgrades_to_inconclusive(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Misleading\n"
                            "Reason: It cannot be determined whether the image visually supports the caption.\n"
                            "Visual Evidence: Blurred street view with unrecognizable objects."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Inconclusive")

    @patch("app.services.openrouter_service.requests.post")
    def test_incidental_detail_unknown_preserves_genuine_verdict(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine\n"
                            "Reason: The domestic cat is clearly lying on the rug as stated; the cat's exact breed is unknown.\n"
                            "Visual Evidence: Tabby cat resting on a living room rug."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Genuine")
        self.assertIn("exact breed is unknown", result["reason"])

    @patch("app.services.openrouter_service.requests.post")
    def test_conflicting_prediction_lines_returns_inconclusive(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine\n"
                            "Verdict: Misleading\n"
                            "Reason: Certain visual elements match the caption while others directly contradict it.\n"
                            "Visual Evidence: Elements are partially consistent and partially contradictory."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Inconclusive")

    @patch("app.services.openrouter_service.requests.post")
    def test_agreeing_multiple_prediction_lines_preserves_verdict(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "openrouter/free",
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine\n"
                            "Verdict: Genuine\n"
                            "Reason: The image clearly depicts a tabby cat sitting on a rug.\n"
                            "Visual Evidence: Tabby cat on carpet."
                        )
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Genuine")

    @patch("app.services.openrouter_service.requests.post")
    def test_compound_verdict_genuine_or_inconclusive_rejected(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine or Inconclusive\n"
                            "Reason: Some elements match but others cannot be verified.\n"
                            "Visual Evidence: Partially matching scene."
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

            err_text = str(ctx.exception)
            self.assertTrue(
                "compound" in err_text.lower() or "did not provide a recognizable prediction" in err_text.lower()
            )
            self.assertNotIn("Genuine", err_text.split("Ambiguous compound prediction:")[0] if "Ambiguous" in err_text else err_text)

    @patch("app.services.openrouter_service.requests.post")
    def test_compound_verdict_slash_misleading_rejected(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine/Misleading\n"
                            "Reason: Mixed signals.\n"
                            "Visual Evidence: Details partially match."
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

            err_text = str(ctx.exception)
            self.assertTrue(
                "compound" in err_text.lower() or "did not provide a recognizable prediction" in err_text.lower()
            )

    @patch("app.services.openrouter_service.requests.post")
    def test_does_not_have_sufficient_evidence_downgrades_to_inconclusive(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Genuine\n"
                            "Reason: The photo does not have sufficient evidence to confirm the location or date.\n"
                            "Visual Evidence: Generic street corner with anonymous pedestrians."
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Inconclusive")

    @patch("app.services.openrouter_service.requests.post")
    def test_misleading_prediction_with_explicit_visual_contradiction_not_downgraded(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Prediction: Misleading\n"
                            "Reason: The image clearly depicts a daytime sunny beach with palm trees, which directly contradicts the caption describing a nighttime blizzard. The external claimed blizzard cannot be verified from this photo.\n"
                            "Visual Evidence: Sunny beach with palm trees and ocean, no snow or blizzard."
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Misleading")

    @patch("app.services.openrouter_service.requests.post")
    def test_numbered_response_lines_parsed_correctly(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "1. Prediction: Genuine\n"
                            "2. Reason: The image clearly depicts a tabby cat sitting on a rug.\n"
                            "3. Visual Evidence: Tabby cat on carpet."
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            result = analyze_remote_image(self.dummy_image, "image/jpeg", self.sample_caption)

        self.assertEqual(result["prediction"], "Genuine")
        self.assertEqual(result["reason"], "The image clearly depicts a tabby cat sitting on a rug.")
        self.assertEqual(result["visual_evidence"], "Tabby cat on carpet.")



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

    def test_api_verify_remote_mode_inconclusive(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                return_value={
                    "prediction": "Inconclusive",
                    "reason": "The image is too blurry to identify the individual.",
                    "visual_evidence": "Obscured figures with heavy motion blur.",
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
                self.assertEqual(data["prediction"], "Inconclusive")
                self.assertEqual(data["confidence_score"], 0.0)
                self.assertEqual(data["confidenceScore"], 0.0)
                self.assertEqual(
                    data["inconsistency_type"],
                    "Remote visual ambiguity / inconclusive",
                )
                self.assertIn("inconclusive or insufficient", data["explanation"])
                self.assertFalse(data["is_stub"])
                self.assertEqual(
                    data["pipeline_stages"][3]["detail"],
                    "Remote model prediction: Inconclusive.",
                )

    def test_api_verify_remote_mode_rate_limit_502(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                side_effect=RuntimeError("Remote model request failed with status 429: Rate limit exceeded"),
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
                self.assertIn("429", data["detail"])

    def test_api_verify_remote_mode_unrecognized_prediction_502(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                side_effect=RuntimeError("The remote model did not provide a recognizable prediction."),
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
                self.assertIn("did not provide a recognizable prediction", data["detail"])

    def test_api_verify_remote_mode_raw_model_text_not_leaked_in_502(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                side_effect=RuntimeError(
                    "The remote model did not provide a recognizable prediction. "
                    "Model response: Sensitive personal user photo details and private names."
                ),
            ):
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/verify",
                        files={"image": ("test.jpg", self.dummy_image, "image/jpeg")},
                        data={"caption": "A person wearing a white shirt"},
                    )

                self.assertEqual(resp.status_code, 502)
                data = resp.json()
                self.assertNotIn("Sensitive personal user photo details", data["detail"])
                self.assertNotIn("Model response", data["detail"])
                self.assertIn("Remote verification failed", data["detail"])
                self.assertIn("did not provide a recognizable prediction", data["detail"])

    def test_api_verify_remote_mode_upstream_provider_html_not_leaked_in_502(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                side_effect=RuntimeError(
                    "Remote model request failed with status 502: "
                    "<html><head><title>502 Bad Gateway</title></head>"
                    "<body>Cloudflare internal server error host=xyz-secret-endpoint</body></html>"
                ),
            ):
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/verify",
                        files={"image": ("test.jpg", self.dummy_image, "image/jpeg")},
                        data={"caption": "A person wearing a white shirt"},
                    )

                self.assertEqual(resp.status_code, 502)
                data = resp.json()
                self.assertNotIn("<html>", data["detail"])
                self.assertNotIn("Cloudflare", data["detail"])
                self.assertNotIn("xyz-secret-endpoint", data["detail"])
                self.assertIn("Remote verification failed", data["detail"])
                self.assertIn("502", data["detail"])

    def test_api_verify_remote_mode_rate_limit_details_not_leaked_in_502(self):
        with patch.dict(os.environ, {"OOC_VERIFY_MODE": "remote", "OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "app.api.routes.verification.analyze_remote_image",
                side_effect=RuntimeError(
                    "Remote model request failed with status 429: "
                    "Rate limit exceeded for account_id=act_987654 balance=0.00 usd"
                ),
            ):
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/verify",
                        files={"image": ("test.jpg", self.dummy_image, "image/jpeg")},
                        data={"caption": "A person wearing a white shirt"},
                    )

                self.assertEqual(resp.status_code, 502)
                data = resp.json()
                self.assertNotIn("act_987654", data["detail"])
                self.assertNotIn("balance=0.00", data["detail"])
                self.assertIn("Remote verification failed", data["detail"])
                self.assertIn("429", data["detail"])

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

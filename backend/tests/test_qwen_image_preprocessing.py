import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from app.services.qwen_service import (
    DEFAULT_MAX_IMAGE_EDGE,
    DEFAULT_MAX_IMAGE_PIXELS,
    prepare_image_for_qwen,
)


class TestQwenImagePreprocessing(unittest.TestCase):
    """
    Focused test suite for Qwen image resolution optimization:
    - Large image downscaling
    - Aspect ratio preservation
    - Small image non-upscaling
    - Boundary condition handling
    - Configurable environment limits
    - Color mode conversions
    - Invalid input handling
    """

    @classmethod
    def setUpClass(cls):
        cls.backend_dir = Path(__file__).resolve().parents[1]
        cls.test_images_dir = cls.backend_dir / "test_images"
        cls.test_image_large = cls.test_images_dir / "test_image2.jpg"
        cls.test_image_small = cls.test_images_dir / "test_image3.jpeg"

    def test_large_image_resizing_test_image2(self):
        """Verify 2340x4160 image is resized below 1024 max edge and 1003520 max pixels."""
        if not self.test_image_large.exists():
            self.skipTest(f"Test image not found: {self.test_image_large}")

        resized_img, orig_dims, resized_dims = prepare_image_for_qwen(self.test_image_large)

        self.assertEqual(orig_dims, (2340, 4160))
        self.assertLessEqual(max(resized_dims), DEFAULT_MAX_IMAGE_EDGE)
        self.assertLessEqual(resized_dims[0] * resized_dims[1], DEFAULT_MAX_IMAGE_PIXELS)
        self.assertEqual(resized_dims, (576, 1024))

        # Check aspect ratio preservation
        orig_ar = orig_dims[0] / orig_dims[1]
        resized_ar = resized_dims[0] / resized_dims[1]
        self.assertAlmostEqual(orig_ar, resized_ar, places=3)

    def test_synthetic_large_landscape(self):
        """Verify synthetic 3000x1500 landscape image is scaled to 1024x512."""
        img = Image.new("RGB", (3000, 1500), color="blue")
        resized_img, orig_dims, resized_dims = prepare_image_for_qwen(img)

        self.assertEqual(orig_dims, (3000, 1500))
        self.assertEqual(resized_dims, (1024, 512))
        self.assertEqual(resized_dims[0] / resized_dims[1], 2.0)

    def test_synthetic_large_square(self):
        """Verify synthetic 2500x2500 square image is clamped by max_pixels constraint."""
        img = Image.new("RGB", (2500, 2500), color="green")
        resized_img, orig_dims, resized_dims = prepare_image_for_qwen(img)

        self.assertEqual(orig_dims, (2500, 2500))
        self.assertLessEqual(max(resized_dims), DEFAULT_MAX_IMAGE_EDGE)
        self.assertLessEqual(resized_dims[0] * resized_dims[1], DEFAULT_MAX_IMAGE_PIXELS)
        # 1001 * 1001 = 1,002,001 <= 1,003,520
        self.assertEqual(resized_dims, (1001, 1001))
        self.assertEqual(resized_dims[0], resized_dims[1])

    def test_small_image_never_upscaled_real_image(self):
        """Verify small image test_image3.jpeg (547x365) retains exact original dimensions."""
        if not self.test_image_small.exists():
            self.skipTest(f"Test image not found: {self.test_image_small}")

        resized_img, orig_dims, resized_dims = prepare_image_for_qwen(self.test_image_small)

        self.assertEqual(orig_dims, (547, 365))
        self.assertEqual(resized_dims, (547, 365))
        self.assertEqual(resized_img.size, (547, 365))

    def test_synthetic_small_image_never_upscaled(self):
        """Verify tiny 150x80 image is never enlarged."""
        img = Image.new("RGB", (150, 80), color="yellow")
        resized_img, orig_dims, resized_dims = prepare_image_for_qwen(img)

        self.assertEqual(orig_dims, (150, 80))
        self.assertEqual(resized_dims, (150, 80))

    def test_exact_boundary_image_not_modified(self):
        """Verify image at exactly 1024x768 is not resized."""
        img = Image.new("RGB", (1024, 768), color="red")
        resized_img, orig_dims, resized_dims = prepare_image_for_qwen(img)

        self.assertEqual(orig_dims, (1024, 768))
        self.assertEqual(resized_dims, (1024, 768))

    def test_configurable_limits_via_parameters(self):
        """Verify explicit max_edge and max_pixels override defaults."""
        img = Image.new("RGB", (1600, 800), color="purple")
        resized_img, orig_dims, resized_dims = prepare_image_for_qwen(
            img, max_edge=400, max_pixels=100000
        )

        self.assertEqual(orig_dims, (1600, 800))
        self.assertEqual(resized_dims, (400, 200))
        self.assertLessEqual(resized_dims[0], 400)
        self.assertLessEqual(resized_dims[0] * resized_dims[1], 100000)

    def test_configurable_limits_via_env(self):
        """Verify environment variables QWEN_MAX_IMAGE_EDGE and QWEN_MAX_IMAGE_PIXELS."""
        img = Image.new("RGB", (2000, 1000), color="orange")
        old_edge = os.environ.get("QWEN_MAX_IMAGE_EDGE")
        old_pixels = os.environ.get("QWEN_MAX_IMAGE_PIXELS")

        try:
            os.environ["QWEN_MAX_IMAGE_EDGE"] = "600"
            os.environ["QWEN_MAX_IMAGE_PIXELS"] = "200000"

            resized_img, orig_dims, resized_dims = prepare_image_for_qwen(img)
            self.assertEqual(resized_dims, (600, 300))
        finally:
            if old_edge is not None:
                os.environ["QWEN_MAX_IMAGE_EDGE"] = old_edge
            else:
                os.environ.pop("QWEN_MAX_IMAGE_EDGE", None)

            if old_pixels is not None:
                os.environ["QWEN_MAX_IMAGE_PIXELS"] = old_pixels
            else:
                os.environ.pop("QWEN_MAX_IMAGE_PIXELS", None)

    def test_rgba_conversion_to_rgb(self):
        """Verify RGBA image with alpha channel is converted to RGB mode."""
        img_rgba = Image.new("RGBA", (800, 600), color=(255, 0, 0, 128))
        resized_img, _, _ = prepare_image_for_qwen(img_rgba)

        self.assertEqual(resized_img.mode, "RGB")

    def test_grayscale_conversion_to_rgb(self):
        """Verify grayscale 'L' image is converted to RGB mode."""
        img_l = Image.new("L", (800, 600), color=128)
        resized_img, _, _ = prepare_image_for_qwen(img_l)

        self.assertEqual(resized_img.mode, "RGB")

    def test_invalid_input_nonexistent_file(self):
        """Verify non-existent file path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            prepare_image_for_qwen("non_existent_image_12345.jpg")

    def test_invalid_input_empty_file(self):
        """Verify zero-byte empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                prepare_image_for_qwen(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_input_corrupted_file(self):
        """Verify non-image text/corrupted bytes raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"NOT_A_VALID_IMAGE_DATA_1234567890")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                prepare_image_for_qwen(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_input_type(self):
        """Verify passing an unsupported type (e.g., int, list) raises TypeError."""
        with self.assertRaises(TypeError):
            prepare_image_for_qwen(12345)

        with self.assertRaises(TypeError):
            prepare_image_for_qwen(["invalid", "path"])


if __name__ == "__main__":
    unittest.main()

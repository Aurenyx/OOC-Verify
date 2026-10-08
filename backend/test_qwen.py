from app.services.qwen_service import analyze_image

IMAGE_PATH = r"news_clippings\experiments\images\test.jpg"

caption = "A person with spectacles"

print("Starting Qwen test...")

result = analyze_image(
    image_path=IMAGE_PATH,
    caption=caption,
)

print("\n==============================")
print("QWEN RESULT")
print("==============================")
print(result)
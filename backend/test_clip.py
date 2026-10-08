import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Use a small pretrained CLIP model
model_name = "openai/clip-vit-base-patch32"

print("Loading CLIP...")
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

# Change this to an image on your computer
image_path = "test_image.jpg"
image = Image.open(image_path).convert("RGB")

captions = [
    "A person standing in front of a building",
    "A person with spectacles",
    "A dog running through a forest"
]

inputs = processor(
    text=captions,
    images=image,
    return_tensors="pt",
    padding=True
)

inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

scores = outputs.logits_per_image[0]

print("\nImage:", image_path)

for caption, score in zip(captions, scores):
    print(f"Caption: {caption}")
    print(f"CLIP score: {score.item():.4f}")
    print()

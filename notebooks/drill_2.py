import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_paths = sorted(os.listdir("test_images/"))
images = [Image.open(f"test_images/{p}") for p in image_paths]

texts = ["a textbook", "a cat", "a closed environment", "a roadsign", "a handheld item"]

with torch.no_grad():
    inputs = processor(images=images, return_tensors="pt", padding=True)
    result = model.get_image_features(pixel_values=inputs["pixel_values"])
    image_features = result.pooler_output

    text_inputs = processor(text=texts, return_tensors="pt", padding=True)
    text_result = model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
    text_features = text_result.pooler_output

# Normalization
l2_images = F.normalize(image_features, p = 2, dim = 1)
l2_text = F.normalize(text_features, p = 2, dim = 1)

# Similarity
similarity = torch.matmul(l2_text, l2_images.T)

# Top-k indices
topk = torch.topk(similarity, 3)

for i, text in enumerate(texts):
    print(f"\n'{text}':")
    for rank, idx in enumerate(topk.indices[i]):
        print(f"  {rank+1}. {image_paths[idx]} (score: {topk.values[i][rank]:.3f})")

# Save embeddings
torch.save(image_features, 'embeddings/image_embeddings.pt')
import os, json, torch
from PIL import Image
from src.search import engine
from transformers import CLIPModel, CLIPProcessor

model, processor = engine.load_model("openai/clip-vit-base-patch32")

with open("data/coco/captions_val2017.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

batch_size = 32
all_embeddings = []
images_list = data["images"]

for i in range(0, len(images_list), batch_size):
    batch_entries = images_list[i:i+batch_size]
    batch_images = [Image.open(f"data/coco/val2017/{e['file_name']}") for e in batch_entries]
    
    with torch.no_grad():
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        result = model.get_image_features(pixel_values=inputs["pixel_values"])
        all_embeddings.append(result.pooler_output)
    
    if i % 320 == 0:
        print(f"Encoded {i}/{len(images_list)}")

all_embeddings = torch.cat(all_embeddings, dim=0)
print(all_embeddings.shape)

filenames = [e["file_name"] for e in images_list]
torch.save(all_embeddings, "embeddings/coco_val_embeddings.pt")
json.dump(filenames, open("embeddings/coco_val_filenames.json", "w"))
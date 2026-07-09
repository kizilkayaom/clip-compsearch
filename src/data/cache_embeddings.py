import json, os, torch
from PIL import Image
from src.search import engine

model, processor = engine.load_model("openai/clip-vit-base-patch32")

all_files = set()
for json_file in os.listdir("data/sugarcrepe/"):
    with open(f"data/sugarcrepe/{json_file}") as f:
        data = json.load(f)
    for key in data:
        all_files.add(data[key]["filename"])

filenames = sorted(list(all_files))

batch_size = 32
all_embeddings = []

for i in range(0, len(filenames), batch_size):
    batch_files = filenames[i:i+batch_size]
    batch_images = [Image.open(f"data/coco/val2017/{f}") for f in batch_files]
    
    with torch.no_grad():
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        result = model.get_image_features(pixel_values=inputs["pixel_values"])
        all_embeddings.append(result.pooler_output)
    
    if i % 320 == 0:
        print(f"Encoded {i}/{len(filenames)}")

all_embeddings = torch.cat(all_embeddings, dim=0)
print(all_embeddings.shape)

torch.save(all_embeddings, "embeddings/demo_gallery_embeddings.pt")
json.dump(filenames, open("embeddings/demo_gallery_filenames.json", "w"))
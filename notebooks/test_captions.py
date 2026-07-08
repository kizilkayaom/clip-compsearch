import os, json
from collections import Counter

with open("data/coco/captions_val2017.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

image_ids = [a["image_id"] for a in data["annotations"]]
counts = Counter(image_ids)
print("Most common:", counts.most_common(3))
print("Least common:", counts.most_common()[-3:])

id_to_caption = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in id_to_caption:
        id_to_caption[img_id] = ann["caption"]

captions = [id_to_caption[img["id"]] for img in data["images"]]
print(captions[:5])

json.dump(captions, open("embeddings/coco_val_captions.json", "w"))
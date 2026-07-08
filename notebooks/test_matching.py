import os, json, torch
from PIL import Image
from src.search import engine
from src.eval.matching import evaluate_file

model, processor = engine.load_model("openai/clip-vit-base-patch32")

path = "data/sugarcrepe/swap_obj.json"

total, accuracy = evaluate_file(model, processor, path)
print(f"Normal:  {accuracy:.4f} (expected ~0.612)")

with open(path, 'r') as f:
    data = json.load(f)

flipped_correct = 0
flipped_total = 0
for key in data:
    entry = data[key]
    image = Image.open(f"data/coco/val2017/{entry['filename']}")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt", padding=True)
        img_feat = model.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
        txt_inputs = processor(text=[entry["caption"], entry["negative_caption"]], return_tensors="pt", padding=True)
        txt_feat = model.get_text_features(input_ids=txt_inputs["input_ids"], attention_mask=txt_inputs["attention_mask"]).pooler_output
    l2_img, l2_txt = engine.normalize(img_feat, txt_feat)
    sim = engine.construct_sim_matrix(l2_txt, l2_img)
    if sim[1] > sim[0]:
        flipped_correct += 1
    flipped_total += 1

print(f"Flipped: {flipped_correct/flipped_total:.4f} (expected ~0.388)")
print(f"Sum:     {accuracy + flipped_correct/flipped_total:.4f} (expected 1.0)")
import json, torch
from PIL import Image
from src.search import engine
from peft import PeftModel
import torch.nn.functional as F

base_model, processor = engine.load_model("openai/clip-vit-base-patch32")

tuned_model, _ = engine.load_model("openai/clip-vit-base-patch32")
tuned_model.text_model = PeftModel.from_pretrained(
    tuned_model.text_model,
    "checkpoints/lora_epoch5"
)

with open("data/sugarcrepe/swap_obj.json", 'r') as f:
    data = json.load(f)

fails = []

for key in data:
    entry = data[key]
    filename = entry["filename"]
    caption = entry["caption"]
    negative = entry["negative_caption"]
    image = Image.open(f"data/coco/val2017/{filename}")

    with torch.no_grad():
        img_inputs = processor(images=image, return_tensors="pt", padding=True)
        base_img = base_model.get_image_features(pixel_values=img_inputs["pixel_values"]).pooler_output
        tuned_img = tuned_model.get_image_features(pixel_values=img_inputs["pixel_values"]).pooler_output

        txt_inputs = processor(text=[caption, negative], return_tensors="pt", padding=True)
        base_txt = base_model.get_text_features(input_ids=txt_inputs["input_ids"], attention_mask=txt_inputs["attention_mask"]).pooler_output
        tuned_txt = tuned_model.get_text_features(input_ids=txt_inputs["input_ids"], attention_mask=txt_inputs["attention_mask"]).pooler_output

    base_sim = torch.matmul(F.normalize(base_txt, dim=1), F.normalize(base_img, dim=1).T)
    tuned_sim = torch.matmul(F.normalize(tuned_txt, dim=1), F.normalize(tuned_img, dim=1).T)

    base_correct = base_sim[0] > base_sim[1]
    tuned_correct = tuned_sim[0] > tuned_sim[1]

    if not base_correct and not tuned_correct:
        margin = (tuned_sim[1] - tuned_sim[0]).item()
        fails.append((margin, filename, caption, negative))

fails.sort(reverse=True)

print(f"Found {len(fails)} cases where fine-tuned fails and base wins\n")
for margin, filename, caption, negative in fails[:10]:
    print(f"File: {filename}")
    print(f"  Correct:  {caption}")
    print(f"  Negative: {negative}")
    print(f"  Margin:   {margin:.4f}")
    print()
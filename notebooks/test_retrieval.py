import json, torch
from src.search import engine
from src.eval.recall_at_k import recall_at_k
from peft import LoraConfig, get_peft_model

model, processor = engine.load_model("openai/clip-vit-base-patch32")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model.text_model = get_peft_model(model.text_model, lora_config)
model.text_model.load_adapter("checkpoints/lora_epoch5", adapter_name="default")

with open("embeddings/coco_val_captions.json", 'r') as f:
    all_captions = json.load(f)

image_embeddings = torch.load("embeddings/coco_val_embeddings.pt")

subset_size = 500
subset_images = image_embeddings[:subset_size]
subset_captions = all_captions[:subset_size]

with torch.no_grad():
    text_inputs = processor(text=subset_captions, return_tensors="pt", padding=True, truncation=True)
    text_result = model.get_text_features(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"]
    )
    finetuned_text = text_result.pooler_output

print(f"Fine-tuned R@1:  {recall_at_k(subset_images, finetuned_text, 1):.4f}")
print(f"Fine-tuned R@5:  {recall_at_k(subset_images, finetuned_text, 5):.4f}")
print(f"Fine-tuned R@10: {recall_at_k(subset_images, finetuned_text, 10):.4f}")

fresh_model, fresh_processor = engine.load_model("openai/clip-vit-base-patch32")

with torch.no_grad():
    fresh_inputs = fresh_processor(text=subset_captions, return_tensors="pt", padding=True, truncation=True)
    fresh_result = fresh_model.get_text_features(
        input_ids=fresh_inputs["input_ids"],
        attention_mask=fresh_inputs["attention_mask"]
    )
    base_text = fresh_result.pooler_output

print(f"\nBase CLIP R@1:   {recall_at_k(subset_images, base_text, 1):.4f}")
print(f"Base CLIP R@5:   {recall_at_k(subset_images, base_text, 5):.4f}")
print(f"Base CLIP R@10:  {recall_at_k(subset_images, base_text, 10):.4f}")
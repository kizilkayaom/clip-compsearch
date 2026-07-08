import json, torch
import torch.nn.functional as F
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
from src.data import negatives
from src.search import engine

model, processor = engine.load_model("openai/clip-vit-base-patch32")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model.text_model = get_peft_model(model.text_model, lora_config)

def infonce_loss(image_embed, text_embed, temp):
    l2_images = F.normalize(image_embed, p=2, dim=1)
    l2_text = F.normalize(text_embed, p=2, dim=1)
    similarities = torch.matmul(l2_images, l2_text.T)
    sim_scaled = similarities / temp
    targets = torch.arange(len(l2_images))
    return F.cross_entropy(sim_scaled, targets)

image_embed = torch.load("embeddings/coco_val_embeddings.pt")
with open("embeddings/coco_val_captions.json", 'r') as f:
    captions = json.load(f)

optimizer = torch.optim.AdamW(model.text_model.parameters(), lr=1e-5)
batch_size = 32
epochs = 5

for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0

    for i in range(0, len(image_embed), batch_size):
        batch_images = image_embed[i:i+batch_size]
        batch_captions = captions[i:i+batch_size]
        batch_negatives = [negatives.swap_objects(negatives.nlp, cap) for cap in batch_captions]

        valid = [(img, pos, neg) for img, pos, neg in
                 zip(batch_images, batch_captions, batch_negatives) if pos != neg]

        if len(valid) < 2:
            continue

        valid_images = torch.stack([v[0] for v in valid])
        valid_positives = [v[1] for v in valid]
        valid_negatives = [v[2] for v in valid]

        combined_texts = valid_positives + valid_negatives

        text_inputs = processor(text=combined_texts, return_tensors="pt", padding=True)
        text_result = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_features = text_result.pooler_output

        optimizer.zero_grad()
        loss = infonce_loss(valid_images, text_features, temp=0.07)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

        if i % 320 == 0:
            print(f"Epoch {epoch}, batch {i}, loss: {loss.item():.4f}")

    print(f"--- Epoch {epoch} avg loss: {epoch_loss/batch_count:.4f} ---")

model.text_model.save_pretrained("checkpoints/lora_epoch5")
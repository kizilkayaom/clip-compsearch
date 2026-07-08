from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

for name, module in model.text_model.named_modules():
    if "proj" in name or "attn" in name:
        print(name)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model.text_model = get_peft_model(model.text_model, lora_config)
model.text_model.print_trainable_parameters()



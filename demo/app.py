import random, json, torch
import gradio as gr
from PIL import Image
from src.search import engine
import torch.nn.functional as F
from peft import LoraConfig, PeftModel

base_model, processor = engine.load_model("openai/clip-vit-base-patch32")

tuned_model, _ = engine.load_model("openai/clip-vit-base-patch32")
tuned_model.text_model = PeftModel.from_pretrained(
    tuned_model.text_model, 
    "checkpoints/lora_epoch5"
)

image_embeddings = torch.load("embeddings/coco_val_embeddings.pt")

with open("embeddings/coco_val_filenames.json", 'r') as f:
    filenames = json.load(f)

with open("data/sugarcrepe/swap_obj.json", 'r') as f:
    sugar_data = json.load(f)

def search(query, k=5):
    with torch.no_grad():
        text_inputs = processor(text=[query], return_tensors="pt", padding=True)
        
        base_feat = base_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        ).pooler_output
        
        tuned_feat = tuned_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        ).pooler_output

        l2_base = F.normalize(base_feat, p=2, dim=1)
        l2_tuned = F.normalize(tuned_feat, p=2, dim=1)
        l2_images = F.normalize(image_embeddings, p=2, dim=1)

        base_sim = torch.matmul(l2_base, l2_images.T)
        tuned_sim = torch.matmul(l2_tuned, l2_images.T)

        base_topk = torch.topk(base_sim, k)
        tuned_topk = torch.topk(tuned_sim, k)

        base_images = [(Image.open(f"data/coco/val2017/{filenames[idx]}"), 
                        f"#{r+1} ({base_topk.values[0][r]:.3f})") 
                       for r, idx in enumerate(base_topk.indices[0])]
        
        tuned_images = [(Image.open(f"data/coco/val2017/{filenames[idx]}"), 
                         f"#{r+1} ({tuned_topk.values[0][r]:.3f})") 
                        for r, idx in enumerate(tuned_topk.indices[0])]

        return (base_images, tuned_images)

def match_example(mode):
    keys = list(sugar_data.keys())
    
    for _ in range(100):
        key = random.choice(keys)
        entry = sugar_data[key]
        image = Image.open(f"data/coco/val2017/{entry['filename']}")
        caption = entry["caption"]
        negative = entry["negative_caption"]

        with torch.no_grad():
            img_inputs = processor(images=image, return_tensors="pt", padding=True)
            base_img = base_model.get_image_features(pixel_values=img_inputs["pixel_values"]).pooler_output
            tuned_img = tuned_model.get_image_features(pixel_values=img_inputs["pixel_values"]).pooler_output

            txt_inputs = processor(text=[caption, negative], return_tensors="pt", padding=True)
            base_txt = base_model.get_text_features(input_ids=txt_inputs["input_ids"], attention_mask=txt_inputs["attention_mask"]).pooler_output
            tuned_txt = tuned_model.get_text_features(input_ids=txt_inputs["input_ids"], attention_mask=txt_inputs["attention_mask"]).pooler_output

        base_sim = torch.matmul(F.normalize(base_txt, dim=1), F.normalize(base_img, dim=1).T)
        tuned_sim = torch.matmul(F.normalize(tuned_txt, dim=1), F.normalize(tuned_img, dim=1).T)

        base_right = base_sim[0] > base_sim[1]
        tuned_right = tuned_sim[0] > tuned_sim[1]

        if mode == "Fine-tune wins" and not base_right and tuned_right:
            break
        elif mode == "Both fail" and not base_right and not tuned_right:
            break
        elif mode == "Both succeed" and base_right and tuned_right:
            break
        elif mode == "Random":
            break

    base_correct = "Pass ✅" if base_right else "Fail ❌"
    tuned_correct = "Pass ✅" if tuned_right else "Fail ❌"

    results = f"""### Correct caption
"{caption}"

### Hard negative (swapped)
"{negative}"

### Scores
| Model | Correct caption | Negative caption | Verdict |
|-------|----------------|-----------------|---------|
| Base CLIP | {base_sim[0].item():.4f} | {base_sim[1].item():.4f} | {base_correct} |
| Fine-tuned | {tuned_sim[0].item():.4f} | {tuned_sim[1].item():.4f} | {tuned_correct} |
"""
    return image, results

with gr.Blocks(title="CLIP-CompSearch") as demo:
    with gr.Tab("Search"):
        gr.Markdown("# CLIP-CompSearch")
        gr.Markdown("A compositional image search engine that handles relational queries standard CLIP gets wrong.")
        with gr.Row():
            query = gr.Textbox(label="Search query", placeholder="a woman preparing pizza while a man watches", scale=4)
            k_slider = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Results", scale=1)
        btn = gr.Button("Search", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base CLIP")
                base_gallery = gr.Gallery(show_label=False, columns=5, height=300)
            with gr.Column():
                gr.Markdown("### Fine-tuned CLIP (LoRA)")
                tuned_gallery = gr.Gallery(show_label=False, columns=5, height=300)
        btn.click(fn=search, inputs=[query, k_slider], outputs=[base_gallery, tuned_gallery])

    with gr.Tab("Matching Test"):
        gr.Markdown("# Compositional Matching")
        gr.Markdown("SugarCrepe examples: does the model score the correct caption higher than the swapped one?")
        with gr.Row():
            mode = gr.Dropdown(choices=["Fine-tune wins", "Both fail", "Both succeed", "Random"], value="Fine-tune wins", label="Show")
            match_btn = gr.Button("Next example", variant="primary")
        with gr.Row():
            match_image = gr.Image(label="Image", height=400)
            match_results = gr.Markdown()
        match_btn.click(fn=match_example, inputs=mode, outputs=[match_image, match_results])

demo.launch()
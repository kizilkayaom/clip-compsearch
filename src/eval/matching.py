import os, json, torch
from PIL import Image
from src.search import engine

model, processor = engine.load_model("openai/clip-vit-base-patch32")

def evaluate_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data_dictionary = json.load(file)

    total = 0
    correct = 0

    for keys in data_dictionary:
        total += 1
        entry = data_dictionary[keys]
        filename = entry['filename']
        caption = entry['caption']
        negative = entry['negative_caption']
        image = Image.open(f"data/coco/val2017/{filename}")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt", padding=True)
            result = model.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = result.pooler_output

            text_inputs = processor(text=[caption, negative], return_tensors="pt", padding=True)
            text_result = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            text_features = text_result.pooler_output

        l2_images, l2_text = engine.normalize(image_features, text_features)
        similarity = engine.construct_sim_matrix(l2_text, l2_images)

        if similarity[0] > similarity[1]:
            correct += 1

    return total, correct / total

rows = []
for json_file in sorted(os.listdir("data/sugarcrepe/")):
    path = f"data/sugarcrepe/{json_file}"
    category = os.path.splitext(json_file)[0]
    total, accuracy = evaluate_file(path)
    rows.append((category, total, accuracy))

header = ("Category", "Total", "Accuracy")
col_width = max(len(header[0]), *(len(r[0]) for r in rows))
print(f"{header[0]:<{col_width}} | {header[1]:>6} | {header[2]:>8}")
print(f"{'-' * col_width}-+-{'-' * 6}-+-{'-' * 8}")
for category, total, accuracy in rows:
    print(f"{category:<{col_width}} | {total:>6} | {accuracy:>8.3f}")
import os, json, torch
from PIL import Image
from src.search import engine

model, processor = engine.load_model("openai/clip-vit-base-patch32")

with open("data/sugarcrepe/swap_obj.json", 'r', encoding='utf-8') as file:
    data_dictionary = json.load(file)

total = 0
correct = 0
wrong = 0
for keys in data_dictionary:
    total +=1
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
        text_result = model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
        text_features = text_result.pooler_output

    l2_images, l2_text = engine.normalize(image_features, text_features)
    similarity = engine.construct_sim_matrix(l2_text, l2_images)
    if similarity[0] > similarity[1]:
        correct += 1
    if similarity[1] > similarity[0]:
        wrong += 1

print(f"Total number of images in swap_obj is {total}: The correctness rate is {correct / total}, and the wrongness rate is {wrong / total}")

    


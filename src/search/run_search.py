import os
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import engine

model_path = "openai/clip-vit-base-patch32"
image_path = "test_images/"
embed_path = "embeddings/image_embeddings.pt"
lst = ["a textbook", "a cat", "a closed environment", "a roadsign", "a handheld item"] 

# One function call to prevent model loading twice
model, processor = engine.load_model(model_path)

# Load image path and features from the cached embedding
image_paths, image_features = engine.load_cached_embeddings(image_path, embed_path)

text_features = engine.load_text_data(model, processor, lst)

# Assigning normalized matrices
l2_images, l2_text = engine.normalize(image_features, text_features)

similarity = engine.construct_sim_matrix(l2_text, l2_images)

topk = engine.extract_topk(similarity)

for i, text in enumerate(lst):
    print(f"\n'{text}':")
    for rank, idx in enumerate(topk.indices[i]):
        print(f"  {rank+1}. {image_paths[idx]} (score: {topk.values[i][rank]:.3f})")

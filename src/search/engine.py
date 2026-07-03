import os
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F


def load_model(model_path:str):
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    return (model, processor)


def load_cached_embeddings(image_path: str, embed_path: str):
    paths = sorted(os.listdir(image_path))
    features = torch.load(embed_path)
    return (paths, features)


# Loading text queries form  a list and construct feature matrix
def load_text_data(model, processor,lst):
    texts = lst
    with torch.no_grad():
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        text_result = model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
        text_features = text_result.pooler_output
    return text_features

# Normalize
def normalize(image_features, text_features, normal_p = 2, mat_dim = 1):
    l2_images = F.normalize(image_features, p = normal_p, dim = mat_dim)
    l2_text = F.normalize(text_features, p = normal_p, dim = mat_dim)
    return (l2_images, l2_text)

# Construct the similarity matrix
def construct_sim_matrix(text, image):
    return torch.matmul(text, image.T)

# Extracting top-k indices
def extract_topk(matrix, k_level = 3):
    return torch.topk(matrix, k = k_level)

import torch
import torch.nn as nn
import torch.nn.functional as F

images = torch.randn(10, 512)

queries = torch.randn(5, 512)

l2_images = F.normalize(images, p= 2, dim= 1)
l2_queries = F.normalize(queries, p = 2, dim= 1)

similarity = torch.matmul(l2_queries, l2_images.T)

results = torch.topk(similarity, 3)

print (f"The shape sim. matrix: {similarity.shape}, and top-3 indices: {results}")
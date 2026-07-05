import torch
import torch.nn.functional as F

def recall_at_k(image_embed, text_embed, K):
    
    if(image_embed.shape != text_embed.shape):
        raise Exception("Matrix dimesion mismatch: Recall@K aborted.")
    
    text_embed = F.normalize(text_embed, p = 2, dim = 1)
    image_embed = F.normalize(image_embed, p = 2, dim = 1)

    similarity = torch.matmul(text_embed, image_embed.T)
    k_count = 0
    
    for i in range(len(similarity)):
        sorted_indices = torch.argsort(similarity[i], descending=True)
        if(i in sorted_indices[:K]):
            k_count += 1
    
    return (k_count / len(similarity))

# Test 1
emb = torch.randn(5, 512)
print("Perfect match R@1:", recall_at_k(emb, emb.clone(), 1))

# Test 2
text_emb = emb.clone()
text_emb[0] = emb[3]
print("Broken pair R@1:", recall_at_k(emb, text_emb, 1))
print("Broken pair R@5:", recall_at_k(emb, text_emb, 5))
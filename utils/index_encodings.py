import faiss
import torch
from pathlib import Path

ENC_PATH = Path("weights/doc_encodings.pth")
INDEX_PATH = Path("weights/faiss_index.faiss")
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
cached_encodings_tensors:torch.Tensor

with open(ENC_PATH, "rb") as f:
    cached_encodings_tensors = torch.load(f, weights_only=True)

dimensions = cached_encodings_tensors.shape[1]
index = faiss.IndexFlatIP(dimensions)

index.add(cached_encodings_tensors.detach().numpy())
print(index.ntotal)
#faiss.write_index(index, str(INDEX_PATH))

q = torch.rand(1, 200)
distances, indices = index.search(q.detach().numpy(), 5)
print(distances)
print(indices)
pass
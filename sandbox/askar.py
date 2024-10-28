import faiss
import numpy as np

index = faiss.read_index("weights/faiss_index.faiss")

if __name__ == "__main__":
    query = np.random.rand(1,200)

    dis, idx = index.search(query, 2)

    print(dis, idx)
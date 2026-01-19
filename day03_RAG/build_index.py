from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 加载向量模型（轻量、靠谱、工业常用）
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. 读取文档
with open("docs.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# 3. 文本向量化
embeddings = embed_model.encode(texts)

# 4. 构建 FAISS 向量索引
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# 5. 保存索引
faiss.write_index(index, "knowledge.index")

print("向量库构建完成，共", len(texts), "条知识")

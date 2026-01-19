from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 加载模型 & 向量库
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
index = faiss.read_index("knowledge.index")

# 2. 原始文本（必须和建库时一致）
with open("docs.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# 3. 用户问题
query = "RAG 是怎么减少大模型幻觉的？"

# 4. 向量化问题
query_vec = embed_model.encode([query])

# 5. 检索 Top-2
D, I = index.search(np.array(query_vec), k=2)

print("🔍 检索结果：")
for idx in I[0]:
    print("-", texts[idx])

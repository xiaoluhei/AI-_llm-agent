import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ===== 1. 加载向量模型 & 向量库 =====
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
index = faiss.read_index("knowledge.index")

with open("docs.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# ===== 2. 用户问题 =====
query = "RAG 是怎么减少大模型幻觉的？"

query_vec = embed_model.encode([query])
_, I = index.search(np.array(query_vec), k=3)

retrieved_text = "\n".join([texts[i] for i in I[0]])

# ===== 3. 构造 RAG Prompt =====
prompt = f"""
你是一个严谨的助手，只能根据提供的资料回答问题，
如果资料中没有答案，就说“资料中没有提到”。

【资料】
{retrieved_text}

【问题】
{query}

【回答】
"""

# ===== 4. 加载本地大模型 =====
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ===== 5. 生成回答 =====
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

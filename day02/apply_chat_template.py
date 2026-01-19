from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

messages = [
    {"role": "system", "content": "你是一名耐心的人工智能老师，擅长用通俗的例子解释复杂概念。"},
    {"role": "user", "content": "请用小学生能听懂的话解释什么是大语言模型。"}
]

# 👉 关键：把“对话”转成模型能读的 prompt
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

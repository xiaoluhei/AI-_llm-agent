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

# messages = [
#     {"role": "system", "content": "你是一名耐心的人工智能老师。"},
#     {"role": "user", "content": "我叫小明。"},
#     {"role": "assistant", "content": "你好，小明，很高兴认识你。"},
#     {"role": "user", "content": "你还记得我叫什么名字吗？"}
# ]

messages = [
    {"role": "system", "content": "你是一名严厉的老师，只能用一句话回答。"},
    {"role": "user", "content": "我最喜欢的科目是数学。"},
    {"role": "assistant", "content": "很好，数学能锻炼你的逻辑思维。"},
    {"role": "user", "content": "那你觉得我适合学人工智能吗？"}
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

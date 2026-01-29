import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型ID
model_name = "Qwen/Qwen3-0.6B"

# 设置设备，优先使用GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("模型和分词器加载完成！")

user_input = ""
model_response = ""
messages = [
        {"role": "system", "content": "You are a helpful assistant but can only use emoji to reply user."}
]

while True: 
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Goodbye. It was nice talking to you.")
        break

    # 准备对话输入
    messages.append({"role": "user", "content": user_input})

    # 使用分词器的模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, 
        enable_thinking=True
    )

    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # print("编码后的输入文本:")
    # print(model_inputs)

    # 使用模型生成回答
    # max_new_tokens 控制了模型最多能生成多少个新的Token
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )

    # 将生成的 Token ID 截取掉输入部分
    # 这样我们只解码模型新生成的部分
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # 解码生成的 Token ID
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("\n模型的回答:")
    print("thinking content:", thinking_content)
    print("content:", content)

    messages.append({"role": "assistant", "content": content})
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用LLM进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ LLM响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HelloAgentsLLM_Local:
    """
    为本书 "Hello Agents" 定制的本地LLM客户端。
    它用于调用本地加载的大语言模型（如Qwen）。
    """
    def __init__(self, model_id: str = None, default_temperature: float = 0.8):
        """
        初始化客户端。加载本地模型。
        """
        self.model_id = model_id or os.getenv("LLM_LOCAL_MODEL_ID", "Qwen/Qwen1.5-0.5B-Chat")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_temperature = default_temperature
        
        try:
            print(f"🔄 正在加载本地模型: {self.model_id}")
            print(f"📱 使用设备: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
            print("✅ 模型和分词器加载完成！")
        except Exception as e:
            raise ValueError(f"❌ 加载模型失败: {e}")

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用本地LLM进行思考，并返回其响应。
        """
        print(f"🧠 本地模型 {self.model_id} 正在生成回答...")
        try:
            # 使用分词器的模板格式化输入
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tie_word_embeddings=False
            )

            # 编码输入文本
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # 使用模型生成回答
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=temperature if temperature > 0 else self.default_temperature,
                do_sample=temperature > 0
            )

            # 截取掉输入部分，只保留新生成的部分
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # 解码生成的 Token ID
            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # print("✅ LLM响应成功:")
            # print(response_text)
            return response_text

        except Exception as e:
            print(f"❌ 调用本地LLM时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        # llmClient = HelloAgentsLLM()
        llmClient = HelloAgentsLLM_Local()
        
        user_input = input("You: ") # e.g. 写一个快速排序算法
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": f"{user_input}"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)

''' >>>
--- 调用LLM ---
🧠 正在调用 xxxxxx 模型...
✅ LLM响应成功:
快速排序是一种非常高效的排序算法...
'''

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

DEFAULT_SYSTEM_PROMT = "你是一個人工智能助手"

from transformers import AutoModelForCausalLM, AutoTokenizer

class HelloAgentsLLM_Local:
    """
    为本书 "Hello Agents" 定制的本地LLM客户端。
    它用于调用本地加载的大语言模型（如Qwen）。
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        初始化客户端。加载本地模型。
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        print(f"🔄 加载本地模型: {self.model_name}")
        print(f"📱 使用设备: {self.model.device}")

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        HelloAgent LLM API, 调用LLM进行思考，并返回其响应。
        """
        print(f"🧠 本地模型 {self.model_name} 正在生成回答...")
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # 编码输入文本
            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            # 使用模型生成回答
            response_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )[0][len(model_inputs.input_ids[0]):].tolist()

            # 解码生成的 Token ID
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    
    # llmClient = HelloAgentsLLM()
    llmClient = HelloAgentsLLM_Local()
    
    user_input = input("You: ")
    messages = [{"role": "user", "content": user_input}]
    
    print("--- 调用LLM ---")
    responseText = llmClient.think(messages)
    print(f"Bot: {responseText}")

''' >>>
--- 调用LLM ---
🧠 正在调用 xxxxxx 模型...
✅ LLM响应成功:
快速排序是一种非常高效的排序算法...
'''

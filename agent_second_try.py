# modified from ./agent_first_try.py

# Tools
from agent_experiment.tools.ToolExecutor import ToolExecutor
from agent_experiment.tools.GetWeather_from_wttrin import get_weather
from agent_experiment.tools.GetAttraction_from_TavilySearch import get_attraction

toolExecutor = ToolExecutor()
toolExecutor.registerTool(
    "get_weather", 
    description="(city: str )查询指定城市的实时天气。", 
    func=get_weather)
toolExecutor.registerTool(
    "get_attraction", 
    description="(city: str , weather: str )根据城市和天气搜索推荐的旅游景点。", 
    func=get_attraction)
available_tools = toolExecutor.getAvailableTools()

AGENT_SYSTEM_PROMPT = f"""
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
{available_tools}

# 回覆說明:
如果你想要調用工具，每次回复的文字末尾必須包含一個 Action，調用工具後將會返回給你工具調用結果，結果可能為正確或錯誤，需要在下輪對話中繼續判斷。如果用戶輸入為空，請勿輸出任何 Action，並提示用戶進行輸入。

# Action 格式要求

Action: [你要执行的具体行动]

具體行動的格式必须是以下之一，其他語句將不被接受：
1. function_name(arg_name= "arg_value")
    代表你的具體行動是調用工具
2. Finish["最终答案"]
    代表你的具體行動是結束任務，並將最終答案的文字填入[]中

# 重要提示:
- 不可有複數個 Action，不可有複數個 Action，不可有複數個 Action！

請開始吧~ /no_think
"""

import re
from agent_experiment.tools.ToolExecutor import CheckToolParameterSatisfied

# TODO 調用工具的解析強烈地與系統提示詞綁定，且目前使用模型會自行產生工具調用結果的幻覺，可能需要查閱已定義好的 Agent 調用方式

# --- 1. 配置LLM客户端 ---
from agent_experiment.LLMClient import HelloAgentsLLM_Local
llm = HelloAgentsLLM_Local(AGENT_SYSTEM_PROMPT)

# --- 2. 初始化 ---

# user_input = input("You: ") # e.g. 写一个快速排序算法
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"

print(f"用户输入: {user_prompt}\n" + "="*40)

observation:str = user_prompt
# --- 3. 运行主循环 ---
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")
        
    # 3.2. 调用LLM进行思考
    llm_output = llm.generate_response(observation)

    # 模型可能会输出多余的Thought-Action，需要截断
    # TODO 擷取多餘輸出對似乎有問題，待 Debug
    # match = re.search(r'(Action:.*?)(?=\n\s*(?:Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    # if match:
    #     truncated = match.group(1).strip()
    #     if truncated != llm_output.strip():
    #         llm_output = truncated
    #         print("已截断多余的 Thought-Action 对")

    print(f"模型输出:\n{llm_output}\n")
    
    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break
    
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)[=:]?\s*"([^"]*)"', args_str)) # 盡可能匹配模型輸出的調用字串

    if not tool_name in available_tools:
        observation = f"错误:未定义的工具 '{tool_name}'"
    else: # 模型正確調用工具
        tool = toolExecutor.getTool(tool_name)
        
        # 檢查 LLM 是否提供了工具函數所需求的參數
        is_satisfied, required_kwargs, error_msg = CheckToolParameterSatisfied(tool, kwargs)
        
        if is_satisfied:
            observation = tool(**required_kwargs) # 運行 tool 並將結果存入觀察
        else:
            observation = error_msg # 返回錯誤結果
        

    # 3.4. 观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)

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

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

import re
from agent_experiment.tools.ToolExecutor import CheckToolParameterSatisfied

# --- 1. 配置LLM客户端 ---
from agent_experiment.LLMClient import HelloAgentsLLM_Local
llm = HelloAgentsLLM_Local()

# --- 2. 初始化 ---

# user_input = input("You: ") # e.g. 写一个快速排序算法
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 3. 运行主循环 ---
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")
    
    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)
    
    # 3.2. 调用LLM进行思考
    # llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    exampleMessages = [
            {"role": "system", "content": f"{AGENT_SYSTEM_PROMPT}"},
            {"role": "user", "content": f"{full_prompt}"}
        ]

    llm_output = llm.think(exampleMessages)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    # TODO 擷取多餘輸出對似乎有問題，待 Debug
    
    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break
    
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)[=:]?\s+"([^"]*)"', args_str)) # 盡可能匹配模型輸出的調用字串

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
        

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)

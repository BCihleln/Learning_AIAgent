# ReAct Agent 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用工具的智能助手

可调用的外部工具如下:
{tools}

每次仅能回覆一组 Thought-Action 对，格式如下：
Thought: 你的思考过程与结论，用于分析问题、拆解任务
Action: {{tool_name}}[{{tool_input}}]

注意事项：
- 请严格按照回覆格式进行回应，不可输出复数对回覆
- 当你收集到足够的资讯，能够回答用户询问时，请于 Action: 中输出 Finish: "问题的最终答案"

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

import re
from LLMClient import HelloAgentsLLM, HelloAgentsLLM_Local
from tools.ToolExecutor import ToolExecutor

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM_Local, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break
            else: 
                print(f"Original Response: \n{response_text}\n")

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"💭 思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[?(.*)\]?", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
            
            print(f"👀 观察: \n{observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

# 示例
if __name__ == "__main__":
    from tools.Search_by_SerpApi import search

    llm = HelloAgentsLLM_Local()

    tool = ToolExecutor()
    tool.registerTool(
        search, 
        "一个网页搜索引擎。当你需要回答有關 即時性資訊 或 進行事實驗證時使用此工具，如：獲取當前時間、即時熱點事件等。")

    agent = ReActAgent(llm, tool, 5)

    # initial_promt = input("You: ")
    # agent.run(initial_promt)

    agent.run("Dell 的最新型號電腦是哪台，基礎硬件設備的型號為何？價格多少？與同價位的其他電腦相比有什麼賣點?")

"""
實驗驗證 ReAct 結構的缺憾
对LLM自身能力的强依赖：
    ReAct 流程的成功与否，高度依赖于底层 LLM 的综合能力。如果 LLM 的逻辑推理能力、指令遵循能力或格式化输出能力不足，就很容易在 Thought 环节产生错误的规划，或者在 Action 环节生成不符合格式的指令，导致整个流程中断。

执行效率问题：
    由于其循序渐进的特性，完成一个任务通常需要多次调用 LLM。每一次调用都伴随着网络延迟和计算成本。对于需要很多步骤的复杂任务，这种串行的“思考-行动”循环可能会导致较高的总耗时和费用。

提示词的脆弱性：
    整个机制的稳定运行建立在一个精心设计的提示词模板之上。模板中的任何微小变动，甚至是用词的差异，都可能影响 LLM 的行为。此外，并非所有模型都能持续稳定地遵循预设的格式，这增加了在实际应用中的不确定性。

可能陷入局部最优：
    步进式的决策模式意味着智能体缺乏一个全局的、长远的规划。它可能会因为眼前的 Observation 而选择一个看似正确但长远来看并非最优的路径，甚至在某些情况下陷入“原地打转”的循环中。
"""
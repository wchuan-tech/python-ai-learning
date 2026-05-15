AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。
# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。
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


import requests

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 请求 api
        response = requests.get(url)
        # 返回状态码
        response.raise_for_status()
        # 获取返回的 json 格式数据
        data = response.json()
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
    
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        return f"Error:Network Problems - {e}"

    except (KeyError,IndexError) as e:
        return f"Error:Analysis Weather Data failure,may be the city name is wrong"

import os
from tavily import TavilyClient

def get_attraction(city: str,weather: str) -> str:
    # 获取配置文件的 API
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error:TAVILY_API_KEY Not Exist"

    # 通过 api_key 访问模型
    tavily = TavilyClient(api_key = api_key)

    query = f"'{city}' 在 '{weather}' 天气下最值得去的旅游景点推荐及其理由"

    try:
        response = tavily.search(
            query = query,
            search_depth = 'basic',
            include_answer = True
        )

        if response.get("answer"):
            return response["answer"]

        # 格式化输出结果
        formatted_results = []
        for result in response.get("results",[]):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "Sorry,do not have relevant visits recommend"

        return "According search,the informations are:\n" + "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error:Have Error of Executing Tavily Search"

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAICompatibleClient:

    def __init__(self,model: str,api_key: str,base_url: str):
        self.model = model
        self.client = OpenAI(
            api_key = api_key,
            base_url = base_url
        )

    def generate(self,prompt: str,system_prompt: str) -> str:
        print("Calling the big language model")
        try:
            messages = [
                {'role': 'system','content': system_prompt},
                {'role': 'user','content': prompt}
            ]
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                stream = False
            )
            answer = response.choices[0].message.content
            print("LLM response successfully")
            return answer
        except Exception as e:
            print(f"Happend wrong of Calling LLM Model:{e}")
            return "Error:Happend wrong of Calling Language model service"

import re

llm = OpenAICompatibleClient(
    model = "deepseek-chat",
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_API_BASE")
    )

user_prompt = "Hello,Please search today's weath of Beijing,According to that to recommend a suitable visit"
prompt_history = [f"user requests: {user_prompt}"]

print(f"user input: {user_prompt}\n" + "="*40)

for i in range(5):
    print(f"--- circle {i + 1} ---\n")

    full_prompt = "\n".join(prompt_history)

    llm_output = llm.generate(
        full_prompt,
        system_prompt = AGENT_SYSTEM_PROMPT
        )

    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|$)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
        
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    action_match = re.search(r"Action:(.*)",llm_output,re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ...Action: ...' 的格式。"
        observation_str = f"Observation:{observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)]",action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"
    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)






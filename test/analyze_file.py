import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key = api_key, base_url="https://api.deepseek.com")

file_path = "D:/Python/test/ai_assistant.py"
with open(file_path,"r",encoding="utf-8") as f:
    code_content = f.read()

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个代码审查专家。"},
        {"role": "user", "content": f"请分析以下代码的逻辑，并给出改进建议：\n\n{code_content}"}
    ]
)

print("--- AI 代码审查报告 ---")
print(response.choices[0].message.content)

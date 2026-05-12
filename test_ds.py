from openai import OpenAI

# 1. 填入你刚才复制的 API Key
client = OpenAI(
    api_key="sk-f93ff599836d4d30b307e170e6d7394c",
    base_url="https://api.deepseek.com"
)

# 2. 发起对话
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个幽默的计算机专业助教。"},
            {"role": "user", "content": "请用一句话评价 Java 和 Python。"}
        ],
        stream=False
    )

    # 3. 打印结果
    print("AI 的回答：")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"出错了：{e}")
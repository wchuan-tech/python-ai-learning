import os
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# 1. 之前写好的读取函数
def get_pdf_text(path):
    reader = PdfReader(path)
    return "".join([p.extract_text() for p in reader.pages])

# 2. 调用 AI
def ask_ai_about_pdf(pdf_path, question):
    text_content = get_pdf_text(pdf_path)
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # 将 PDF 内容作为上下文（Context）喂给 AI
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"你是一个专业的学术助教。以下是参考资料内容：\n\n{text_content[:4000]}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# 3. 实战运行
path = r"D:\Python\test\bitcoin.pdf"
print(ask_ai_about_pdf(path, "请用三句话总结这份文档的核心内容。"))
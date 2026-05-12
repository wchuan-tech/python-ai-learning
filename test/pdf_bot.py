from pypdf import PdfReader

def get_pdf_text(path):
    print(f"--- 开始读取文件: {path} ---")
    reader = PdfReader(path)
    print(f"检测到总页数: {len(reader.pages)}") # 看看能不能读到页数
    
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        print(f"第 {i+1} 页提取到的字符数: {len(page_text) if page_text else 0}")
        if page_text:
            text += page_text
    return text

content = get_pdf_text(r"D:\Python\test\study.pdf")

print("\n--- 最终结果 ---")
if not content.strip():
    print("结果：没读到任何文字。这通常说明该 PDF 是图片构成的。")
else:
    print(f"总计提取字数: {len(content)}")
    print(content[:500])
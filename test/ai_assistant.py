import os
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# langsmith
from langsmith.wrappers import wrap_openai

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 加载 .env 配置文件，将 API Key 注入系统环境变量，避免硬编码，保证安全
load_dotenv()

@st.cache_data # Streamlit 的装饰器
# 作用：缓存函数的执行结果。如果上传的是同一个文件，它会直接从内存拿数据，不重新解析 PDF。
def get_pdf_text_from_upload(path):
    """
    参数: uploaded_file 是 Streamlit 提供的 BytesIO 对象（文件流）
    作用: 解析 PDF 二进制数据并提取文字
    """
    reader = PdfReader(path)
    # 列表推导式：遍历每一页，提取文字，最后用空字符串连接成一个大长串
    return "".join([p.extract_text() for p in reader.pages])


# 2. 调用 AI
def ask_ai_with_memory(context, history_messages):

    # 从环境变量读取 Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    # 初始化客户端，base_url 指向 DeepSeek 服务器地址
    base_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    client = wrap_openai(base_client)
    
    system_message = {
        "role": "system",
        "content": f"你是一个专业的学术助教。请基于以下参考资料回答：\n\n{context[:4000]}"
    }

    full_messages = [system_message] + history_messages

    # 调用对话生成接口
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=full_messages
    )
    # 返回 AI 回复的文本内容
    return response.choices[0].message.content



# --- UI 层 ---

# --- 初始化变量 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 设置网页标签页标题
st.set_page_config(page_title = "AI PDF Assistant")
st.title("🤖 智能 PDF 助教")

# 在代码最开始，初始化一个用于控制文件上传器的 key
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# 使用 with 语法定义侧边栏容器
with st.sidebar:
    st.markdown("### ⚙️ 设置")

    # 使用 popover 创建一个气泡容器
    with st.popover("🗑️ 清空对话"):
        st.write("确定要删除所有聊天记录吗？")
        if st.button("确定清空", type="primary"): # type="primary" 会让按钮变红/醒目
            st.session_state.messages = []
            # 改变 key 的值！这会强制 file_uploader 销毁并重建
            st.session_state.file_uploader_key += 1
            st.rerun()

    st.info("1. 上传 PDF\n2. 在下方输入问题\n3. 获取 AI 深度分析")

# 文件上传组件，type=["pdf"] 限制只能上传 PDF
uploaded_file = st.file_uploader(
    "点击或拖拽上传 PDF 文件", 
    type="pdf",
    key=f"uploader_{st.session_state.file_uploader_key}" # 动态绑定 key key 变了 → 组件认为是新组件 → 自动清空、重置！
)

# 【条件渲染】只有当用户上传了文件，下面的逻辑才会运行
if uploaded_file:
    st.info("文件已就绪，请在下方输入问题开始对话 👇")
    with st.spinner("正在解析 PDF,请稍候..."):
        text_content = get_pdf_text_from_upload(uploaded_file)
    
    
    # [渲染记忆] 把之前聊过的内容显示在网页上
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("针对这份文档问我点什么...")

    if question:
        # st.chat_message("user")：自动生成一个带头像的用户对话气泡
        with st.chat_message("user"):
            # 将用户的提问显示在页面中
            st.write(question)
        
        # 将问题存入“记忆列表”
        st.session_state.messages.append(
            {"role": "user", "content": question}
            )

        # 自动生成一个带头像的助手对话气泡
        with st.chat_message("assistant"):
            # 加载等待动画
            with st.spinner("正在思考..."):
                # 调用 AI 函数并获取结果
                answer = ask_ai_with_memory(text_content, st.session_state.messages)
                st.write(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                    )

print(f"DEBUG: LangChain Key exists: {bool(os.getenv('LANGCHAIN_API_KEY'))}")
print(f"DEBUG: Project Name: {os.getenv('LANGCHAIN_PROJECT')}")
from dotenv import load_dotenv

import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

# 核心组件
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

# 免费本地模型导入
from langchain_huggingface import HuggingFaceEmbeddings

import logging
# 屏蔽 transformers 库产生的非错误日志
logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. 修正：必须加括号调用函数
load_dotenv()

# --- 逻辑层 ---
# 创建向量数据库
@st.cache_resource
def build_vector_store(uploader_file):
    """
    Load -> Split -> Embed -> Store 流程 
    """
    # wb 二进制写入
    with open("temp.pdf", "wb") as f:
        f.write(uploader_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # 文件切分格式
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 存入切分后的数据
    splits = text_splitter.split_documents(docs)

    # 【关键精进】换成免费的本地模型
    # 第一次运行会自动下载模型（约 80MB），之后全本地运行，不花钱，没网络限制
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 存入向量数据库（Chroma）
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


def ask_ai_with_rag(vectorstore, question, history):
    # 检索最相关的 3 个片段 retriever：检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    # 把 3 段内容拼接成一段文本
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base="https://api.deepseek.com",
        streaming=True # 开启流式输出
    )

    system_prompt = f"""你是一个专业的学术助教。请仅根据以下参考资料回答问题。参考资料：{context_text}"""

    # 构造 LangChain 标准消息格式
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    # SystemMessage 系统指令
    messages = [SystemMessage(content=system_prompt)]

    # 将 history 转换为对象格式
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    messages.append(HumanMessage(content=question))

    return llm.stream(messages)

# --- UI 表现层 ---
import streamlit as st
st.set_page_config(page_title="标准 RAG 助手")
st.title("🚀 工业级 RAG PDF 助手")

# 初始化变量
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("配置")
    uploaded_file = st.file_uploader("上传 PDF 知识库", type="pdf")
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

if uploaded_file:
    # 构建知识库
    with st.spinner("正在构建本地向量库（首次运行需下载模型）..."):
        vectorstore = build_vector_store(uploaded_file)
    st.sidebar.success("向量数据库已就绪！")

    # 展示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 聊天输入
    question = st.chat_input("基于文档提问...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            # 使用 st.write_stream 实现打字机效果（Streamlit 官方推荐）
            response_generator = ask_ai_with_rag(vectorstore, question, st.session_state.messages[:-1])
            
            # 注意：ask_ai_with_rag 现在返回的是生成器
            # st.write_stream 专门用来流式输出内容（逐段实时显示）
            full_response = st.write_stream(response_generator)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("请先在左侧上传 PDF 文件以启动 RAG 引擎。")
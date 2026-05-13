import os
import logging
import hashlib
import streamlit as st
from dotenv import load_dotenv

# --- 核心库导入 ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Agent 专属导入
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub

# 初始化设置
st.set_page_config(page_title="AI Agent Assistant", layout="wide")
load_dotenv()
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- 【修改点 2】预定义 LLM 对象 ---
# 作为 Agent 的“大脑”，它需要在全局范围内可被访问
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    streaming=True
)

DB_PARENT_DIR = "./chroma_db"

@st.cache_resource
def build_vector_store(uploaded_file):
    """保持不变：Load -> Split -> Embed -> Store (持久化)"""
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    persist_path = os.path.join(DB_PARENT_DIR, file_hash)

    if os.path.exists(persist_path):
        st.sidebar.info("Found Cache, Loading...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=persist_path, embedding_function=embeddings)

    uploaded_file.seek(0)
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_path)

# --- UI 表现层 ---

st.title("🤖 智能 PDF 助手 (Agent 版)")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload PDF file...", type="pdf")
    if st.button("🗑️ Clean Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 动态创建 Agent 及其工具箱 ---
# 专业逻辑：工具必须拿到 vectorstore 后才能被创建
# 工具 A: 联网搜索
search_tool = TavilySearchResults(k=3)
tools = [search_tool]
if uploaded_file:
    # 1. 解析 PDF
    with st.spinner("Building VectorStore..."):
        vectorstore = build_vector_store(uploaded_file)
    st.sidebar.success("Ready!")

    # 工具 B: PDF 检索（将 RAG 变成一个工具）
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "pdf_search",
        "用于搜索当前上传的 PDF 文档。当你需要查找文档里的具体细节时，请使用此工具。"
    )
    
    tools.append(retriever_tool)

# 获取官方 Agent 提示词模板（定义了如何使用工具）
agent_prompt = hub.pull("hwchase17/openai-tools-agent")
    
# 创建 Agent 核心逻辑
agent = create_openai_tools_agent(llm, tools, agent_prompt)
    
# 创建执行器（这是 Agent 的中枢神经系统）
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 【修改点 4】聊天处理逻辑 ---
if question := st.chat_input("问问 PDF 或搜搜全网..."):
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.status("Agent 正在思考并选择工具...", expanded=True) as status:
            # 调用 Agent 执行器
            # 注意：Agent 默认不支持原生 stream，这里我们获取最终 output
            response = agent_executor.invoke({
                "input": question,
                "chat_history": st.session_state.messages[:-1] # 传入记忆
            })
            status.update(label="思考完成！", state="complete", expanded=False)
                
        full_response = response["output"]
        st.write(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Upload a PDF or Search via AI Agent.")
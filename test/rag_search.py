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

# --- 预定义 LLM 对象 ---
# Agent 的“大脑”
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    streaming=True
)

DB_PARENT_DIR = "./chroma_db/global_store"

@st.cache_resource
def build_vector_store(uploaded_files):
    """Load -> Split -> Embed -> Store"""
    all_final_splits = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        with open(temp_path,"wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600,chunk_overlap = 100)
        splits = text_splitter.split_documents(docs)

        for s in splits:
            s.metadata["file_name"] = file.name

        all_final_splits.extend(splits)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return Chroma.from_documents(
        documents=all_final_splits,
        embedding=embeddings,
        persist_directory=DB_PARENT_DIR
    )

# --- UI 表现层 ---

st.title("🤖 智能 PDF 助手 (Agent 版)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 创建 Agent 及其工具箱 ---
# 专业逻辑：工具必须拿到 vectorstore 后才能被创建
# 工具 A: 联网搜索
search_tool = TavilySearchResults(k=3)
tools = [search_tool]

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader("Upload PDF file...", type="pdf",accept_multiple_files=True)
    if st.button("🗑️ Clean Chat"):
        st.session_state.messages = []
        st.rerun()

if uploaded_files:
    # 1. 解析 PDF
    with st.spinner("Building VectorStore..."):
        vectorstore = build_vector_store(uploaded_files)
    st.sidebar.success("Ready!")

    # 工具 B: PDF 检索（将 RAG 变成一个工具）
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(search_kwargs={"k": 5}),
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

# --- 聊天处理逻辑 ---
if question := st.chat_input("问问 PDF 或搜搜全网..."):
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append(
        {"role": "user", "content": question}
        )

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
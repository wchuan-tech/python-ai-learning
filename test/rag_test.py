import os
import logging
import streamlit as st
from dotenv import load_dotenv

# 1. 确保在 import 沉重库之前先显示页面标题（减少白屏焦虑）
st.set_page_config(page_title="Simple RAG Assistant", layout="wide")

# 2. 加载环境变量 (修正括号)
load_dotenv()

# 3. 导入核心库（这些库加载很慢，请耐心等待）
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 屏蔽警告
logging.getLogger("transformers").setLevel(logging.ERROR)

import hashlib
# --- 逻辑层 ---

# 设置持久化根目录
DB_PARENT_DIR = "./chroma_db"

@st.cache_resource
def build_vector_store(uploaded_file):
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest() # 生成唯一的 32 位字符串
    persist_path = os.path.join(DB_PARENT_DIR,file_hash)

    if os.path.exists(persist_path):
        st.sidebar.info("Index Existing,Loading...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory = persist_path, embedding=embeddings)
        
        return vectorstore

    st.sidebar.warning("New File,Constructing Index...")

    uploaded_file.seek(0)
    with open("temp.pdf","wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
        )

    all_splits = text_splitter.split_documents(docs)
    # 第一次运行会在这里下载模型，请观察终端进度条
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=embeddings,
        persist_directory=persist_path # 数据将存入以文件哈希命名的文件夹
        )
    
    return vectorstore

def ask_ai_with_rag(vectorstore, question, history):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 默认会自动读取 .env 里的 OPENAI_API_KEY / DEEPSEEK_API_KEY
    llm = ChatOpenAI(
        model='deepseek-chat',
        streaming=True
    )

    system_prompt = f"你是一个专业的学术助教。请根据参考资料回答问题。资料：{context}"
    messages = [SystemMessage(content=system_prompt)]
    
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
            
    messages.append(HumanMessage(content=question))

    return relevant_docs,llm.stream(messages)

# --- UI 表现层 ---

st.title("🚀 Deployed RAG PDF Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload file...", type="pdf")
    if st.button("Clean Chat"):
        st.session_state.messages = []
        st.rerun()

if uploaded_file:
    with st.spinner("Constructing Local VectorStore (Downloading models if first time)..."):
        vectorstore = build_vector_store(uploaded_file)
    st.sidebar.success("Ready!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask about the PDF..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            sources,response_generator = ask_ai_with_rag(vectorstore, question, st.session_state.messages[:-1])
            full_response = st.write_stream(response_generator)

            with st.expander("🔍 查看参考来源"):
                for i, doc in enumerate(sources):
                    # 从 metadata 中提取页码（PyPDFLoader 自动添加的）
                    page_num = doc.metadata.get("page", "未知") + 1 # +1 是因为页码从 0 开始
                    st.markdown(f"**来源 {i+1} (第 {page_num} 页):**")
                    st.caption(doc.page_content) # 用小字显示原文片段
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("👈 Please upload a PDF file to start.")
import os
from dotenv import load_dotenv

# --- 核心组件导入 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 注意这里：RetrievalQA 现在位于 langchain.chains 模块中
from langchain.chains import RetrievalQA 

# 加载环境变量
load_dotenv()

# 1. 加载文档
loader = PyPDFLoader("doc.pdf")
documents = loader.load()

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3. 创建向量数据库
embeddings = OpenAIEmbeddings(
    openai_api_base="https://api.deepseek.com",
    openai_api_key=os.environ["DEEPSEEK_API_KEY"]
)
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. 创建问答链
# 注意：这里使用的是 langchain 内置的 RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com",
        openai_api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0
    ),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. 交互测试
print("=== 文档问答机器人已启动，输入 '退出' 结束 ===\n")
while True:
    question = input("你：")
    if question.lower() in ["退出", "quit", "exit"]:
        break
    # 注意：新版本 invoke 的参数通常是 "input" 或 "query"，取决于 chain_type
    result = qa_chain.invoke({"query": question})
    print(f"AI：{result['result']}\n")
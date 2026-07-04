import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()

# 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",      # 通用对话模型；也可用 "deepseek-reasoner"（推理更强）
    temperature=0.7,
    max_tokens=2048,
)

# 直接调用
response = llm.invoke("请用100字介绍你自己")
print(response.content)
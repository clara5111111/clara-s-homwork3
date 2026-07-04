import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# 加载环境变量
load_dotenv()

# 1. 定义搜索工具
search = DuckDuckGoSearchRun()
tools = [search]

# 2. 初始化 DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_base="https://api.deepseek.com",
    openai_api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0
)

# 3. 创建 Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个智能问答助手。你可以使用工具来获取最新信息。请根据用户的问题，决定是否需要搜索，然后给出准确、详细的回答。"
)

# 4. 测试运行
print("=== 智能问答 Agent 已启动，输入 '退出' 结束 ===\n")
while True:
    question = input("你：")
    if question.lower() in ["退出", "quit", "exit"]:
        break
    
    # 新版调用方式：传入 messages 列表
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    
    # 提取最终回复（取最后一条消息的内容）
    answer = result["messages"][-1].content
    print(f"AI：{answer}\n")
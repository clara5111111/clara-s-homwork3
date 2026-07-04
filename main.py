import os
from dotenv import load_dotenv

load_dotenv()

print("=== 请选择要运行的 Agent ===")
print("1. 智能问答 Agent")
print("2. 内容创作 Agent")
print("3. 文档问答机器人")

choice = input("\n请输入数字（1/2/3）：")

if choice == "1":
    from qa_agent import agent_executor
    while True:
        question = input("\n你：")
        if question.lower() in ["退出", "quit", "exit"]:
            break
        result = agent_executor.invoke({"input": question})
        print(f"AI：{result['output']}")

elif choice == "2":
    from content_agent import create_article
    topic = input("\n请输入创作主题：")
    article = create_article(topic)
    print("\n" + "="*50)
    print(article["content"])

elif choice == "3":
    from rag_agent import qa_chain
    while True:
        question = input("\n你：")
        if question.lower() in ["退出", "quit", "exit"]:
            break
        result = qa_chain.invoke({"query": question})
        print(f"AI：{result['result']}")

else:
    print("无效选择！")
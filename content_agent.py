import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_base="https://api.deepseek.com",
    openai_api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0.7
)

# 三步链式创作
title_prompt = ChatPromptTemplate.from_template(
    "根据以下主题，生成3个吸引人的标题，用换行分隔：\n主题：{topic}"
)
title_chain = title_prompt | llm

outline_prompt = ChatPromptTemplate.from_template(
    "根据以下标题，生成一篇详细文章大纲（包含引言、正文各章节、结语）：\n标题：{title}"
)
outline_chain = outline_prompt | llm

content_prompt = ChatPromptTemplate.from_template(
    """根据以下大纲，撰写一篇完整的文章，要求：
- 语言流畅自然，避免AI腔
- 每个章节不少于200字
- 适当使用小标题和列表

大纲：
{outline}"""
)
content_chain = content_prompt | llm

def create_article(topic):
    print("📝 正在生成标题...")
    title = title_chain.invoke({"topic": topic}).content
    print(f"✅ 标题：{title}\n")
    
    print("📋 正在生成大纲...")
    outline = outline_chain.invoke({"title": title}).content
    print(f"✅ 大纲：\n{outline}\n")
    
    print("✍️ 正在撰写正文...")
    content = content_chain.invoke({"outline": outline}).content
    print("✅ 文章完成！\n")
    
    return {"title": title, "outline": outline, "content": content}

# 测试
if __name__ == "__main__":
    topic = input("请输入创作主题：")
    article = create_article(topic)
    print("\n" + "="*50)
    print(article["content"])
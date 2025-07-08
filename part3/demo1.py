# 模型IO
# LLM模型：直接传字符串，适合单轮问答。
# Chat模型：传消息对象，适合多轮对话。

from langchain_community.llms import Ollama

# LLM模型调用
llm = Ollama(model="deepseek-r1:7b", temperature=0)
response = llm("你好，介绍一下你自己。")
print("模型响应:", response)

# Chat模型调用
print("\n=== Chat模型调用 ===")
from langchain_community.chat_models import  ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

chat = ChatOllama(model="deepseek-r1:7b", temperature=0)
response = chat.invoke([
    HumanMessage(content="你好，介绍一下你自己。"),
    AIMessage(content="我是一个AI助手，可以回答你的问题。")
])
print("Chat模型响应:", response.content)
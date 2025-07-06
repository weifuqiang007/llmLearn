# 记忆组件
# 基于历史数据进行问答处理
from uuid import uuid4

from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder)
from langchain.chains import conversation
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 存储对话历史的字典（生产环境中建议使用数据库）
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:

    if session_id not in store:
        # store[session_id] = ConversationBufferMemory(return_messages=True, memory_key="history", input_key="input", output_key="history")
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("以下是一段友好的对话，在AI和人类之间。AI助手将根据对话上下文提供信息。如果不知道答案，它将说'我不知道'。"),
    MessagesPlaceholder(variable_name="history"),  # 用于插入对话历史
    HumanMessagePromptTemplate.from_template("{input}")
])

# 创建ollama模型实例
llm = Ollama(model="deepseek-r1:7b", temperature=0)

# 创建基础链
chain = prompt | llm

# 创建一个对话链。这个组件会使用之前创建的llm 和 memory
conversation = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")

# # 定义会话ID（实际应用中应该为每个用户/会话生成唯一ID）
SESSION_ID = str(uuid4())

print("=== 第一次对话 ===")
response = conversation.invoke({
    "input": "你好，AI助手！我给你取一个好听且响亮的名字吧。你就叫做神龙战士！"
},
    config={"configurable": {"session_id": SESSION_ID}}  # 必须提供session_id
)
print("第一次响应:", response)
# 继续对话  
print("\n=== 第二次对话 ===")
response = conversation.invoke(
    {"input": "我之前问了你什么问题？你叫什么名字？"},
    config={"configurable": {"session_id": SESSION_ID}}  # 使用相同的session_id以保持对话连续性
)
print(response)
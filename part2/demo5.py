# LangChain表达式
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder)

# 实例化模板
promet = ChatPromptTemplate.from_template("TELL ME A JOKE ABOUT {topic}")
# 定义模型
llm = Ollama(model="deepseek-r1:7b", temperature=0.7)

# 定义处理链
chain = promet | llm.bind(
    topic=HumanMessagePromptTemplate.from_template("{topic}"),
    system=SystemMessagePromptTemplate.from_template("你是一个有趣的AI助手，请用中文回答"),
    stop={"\n"}
)   

# 使用invoke方法调用模型
response = llm.invoke(promet.format(topic="Python"))    
print(response)
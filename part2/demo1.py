from langchain_ollama import OllamaLLM as Ollama

# 初始化 - 建议添加常用参数
llm = Ollama(
    model="deepseek-r1:7b",
    temperature=0.8,
    top_k=50,
    system="你是一个有帮助的AI助手，请用中文回答"
)

# 使用 invoke 替代 predict
response = llm.invoke(
    "给公司起一个好听的名字，我们是一家做AI的公司，位置在新加坡"
)
print(response)
# 使用 invoke 方法可以直接获取响应内容
# response.content  
print("===========================")
print(response.content)
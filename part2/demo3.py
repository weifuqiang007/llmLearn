# 链
# 将提示词模板和LLM结合起来使用
# 使用LangChain的LLMChain 对模型进行包装，实现与提示词模板类似的功能

from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)

# 定义系统提示词模板
template = (" 你是一个很有帮助的AI助手，尤其是在将{input_language}翻译为{output_language}领域。")
system_message_prompt  = SystemMessagePromptTemplate.from_template(template)

# 定义用户输入模板
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 组合提示词
# ChatPromptTemplate 是专门设计用来组合多个消息模板的类，而 ChatMessagePromptTemplate 是用于单个消息模板的
# format_messages() 会返回一个消息对象列表（包含消息类型和内容），而 format() 只是简单地拼接字符串
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt , human_message_prompt])

# 格式化提示词
formatted_prompt = chat_prompt.format(
    input_language="中文",
    output_language="英文",
    text="请将下面的中文翻译成英文：我爱编程"
)

print("=== 完整提示词 ===")
print(formatted_prompt)
print("\n=== 预期输出 ===")
print("I love programming")

# 初始化模型
llm = Ollama(model="deepseek-r1:7b", temperature=0.7)

# 转换为字符串格式（如果需要）
# prompt_str = "\n".join([msg for msg in formatted_prompt])

# print("\n=== 格式化后的提示词 ===")
# print(prompt_str)
# 调用模型
response = llm.invoke(formatted_prompt)
print("\n=== 模型响应 ===")
print(response)

from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
# 创建链

chain = (
    {"input_language": RunnablePassthrough(), "output_language": RunnablePassthrough(), "text": RunnablePassthrough()} 
    | chat_prompt
    | llm
)

# 弃用写法
# chain = LLMChain(
#     llm=llm,
#     prompt=chat_prompt
# )

# 调用链
# response_chain = chain.run(input_language="中文",
#                            output_language="英文",
#                            text="请将下面的中文翻译成英文：我爱编程")

response_chain = chain.invoke({"input_language" : "中文",
                               "output_language" : "英文",  
                              "text": "请将下面的中文翻译成英文：我爱编程"} )
print("\n=== 链响应 ===")   
print(response_chain)

from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# 用 deepseek 本地模型
llm = Ollama(model="deepseek-r1:7b")

# 示例工具
def search_tool(query):
    return f"假装在搜索: {query}"

tool = Tool(
    name="Search",
    func=search_tool,
    description="用于搜索信息的工具"
)

# 初始化 Agent
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    query = "请帮我搜索一下LangChain是什么。"
    response = agent.run(query)
    print(response)


"""
> Entering new AgentExecutor chain...
<think>
Okay, so I need to figure out what LangChain is. I'm not exactly sure, but I remember it's related to AI or machine learning. Let me start by breaking down the name. "Lang" probably stands for language, and "Chain" might refer to a chain of something, maybe processes or data flow.

I think LangChain could be an open-source tool in AI. Maybe it helps with language processing tasks? I've heard about frameworks like TensorFlow and PyTorch for machine learning, but I'm not sure if LangChain is part of that. It could be used to build applications that handle text, perhaps chatbots or summarization tools.

I should search for "LangChain" to get more information. That way, I can find out its official documentation or GitHub repository to understand how it's structured and what features it offers.
</think>

LangChain is an open-source framework designed to simplify the development of large language models (LLMs). It provides a flexible and extensible architecture for integrating various language processing components, such as text generation, information extraction, and conversational flows. The framework supports building intelligent applications by enabling users to chain together different AI capabilities seamlessly.

**Final Answer:**
LangChain is an open-source framework that facilitates the development of large language models, providing a flexible architecture for creating intelligent applications through the chaining of various AI capabilities.

> Finished chain.
**
LangChain is an open-source framework that facilitates the development of large language models, providing a flexible architecture for creating intelligent applications through the chaining of various AI capabilities.
"""


"""
这是 LangChain Agent 和直接调用 LLM 的核心区别：

直接调用 LLM
你只是把问题发给大模型，模型只能用自己的知识回答，无法调用外部工具，比如搜索、计算、查数据库等。

Agent
Agent 不仅能用大模型，还能根据你的问题自动决定是否调用外部工具（如你定义的 search_tool），并把工具返回的结果和 LLM 的推理结合起来，给你更强大的答案。

流程：

Agent 先用 LLM理解你的问题。
判断是否需要用工具（如搜索）。
调用工具，拿到结果。
再用 LLM 组织最终答案。
总结：

直接 LLM：只能“想”，不能“做”。
Agent：能“想”，还能“做事”（用工具），更智能。
你可以试试问 Agent 一些需要用工具才能回答的问题，体会区别！
"""

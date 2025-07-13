from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm_chain = LLMChain(llm=Ollama(model="deepseek-r1:7b"), prompt=PromptTemplate.from_template("请帮我搜索一下{query}。"))

from  langchain.agents import AgentType
# 用 deepseek 本地模型
llm = Ollama(model="deepseek-r1:7b", temperature=0)
# 加载使用工具。请注意，llm-math工具使用了一个大语言模型接口，所i有需要传递llm参数
tools = load_tools(["serpapi","llm-math"], llm=llm)
# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    query = "请帮我搜索一下LangChain是什么。"
    response = agent.run(query)
    print(response)
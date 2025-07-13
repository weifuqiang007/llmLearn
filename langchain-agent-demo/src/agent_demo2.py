from langchain_community.llms.ollama import Ollama
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
import json
import re

# 自定义搜索工具实现
class CustomSearchTool:
    def __init__(self) :
        pass        
    def _clean_text(self, text):
        """清除HTML标签和多余空白"""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _get_page_content(self, url):
        """获取网页内容"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除script, style等不需要的元素
            for element in soup(['script', 'style', 'iframe', 'noscript']):
                element.decompose()
            
            return self._clean_text(soup.get_text())
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def search(self, query:str) -> str:
        """实现真实网络搜索功能"""
        try:
            from duckduckgo_search import ddg
            results = ddg(query, max_results=5)
            return "\n".join([
                f"标题: {r['title']}\n链接: {r['link']}\n摘要: {r['body']}"
                for r in results
            ])
        # try:
            # 1. 使用模拟API或真实API获取搜索结果
            # 这里使用DuckDuckGo作为示例 (实际需要安装duckduckgo-search包)
            # from duckduckgo_search import ddg
            # results = ddg(query, max_results=self.max_results)
            
            # 示例模拟数据 (实际使用时替换为上面的真实API调用)
            results = [
                {
                    "title": "LangChain 官方文档", 
                    "url": "https://python.langchain.com/docs/get_started/introduction",
                    "snippet": "LangChain是一个用于构建AI应用的框架..."
                },
                {
                    "title": "GitHub - LangChain", 
                    "url": "https://github.com/langchain-ai/langchain",
                    "snippet": "LangChain是一个强大的LLM应用开发框架..."
                }
            ]
            
            # 2. 获取页面内容 (可选)
            enriched_results = []
            for result in results:
                if len(enriched_results) >= self.max_results:
                    break
                
                content = self._get_page_content(result["url"])
                if content:
                    # 截取前500字防止内容过长
                    result["content"] = content[:500] + "..." if len(content) > 500 else content
                    enriched_results.append(result)
            
            # 3. 格式化最终输出
            formatted_output = []
            for idx, result in enumerate(enriched_results, 1):
                formatted_output.append(
                    f"【结果 {idx}】\n"
                    f"标题: {result.get('title', '无标题')}\n"
                    f"链接: {result.get('url', '无链接')}\n"
                    f"摘要: {result.get('snippet', '无摘要')}\n"
                    f"内容: {result.get('content', '无内容')}\n"
                )
            
            return "\n\n".join(formatted_output) if formatted_output else "未找到相关信息"
        
        except Exception as e:
            print(f"搜索出错: {str(e)}")
            return f"搜索失败: {str(e)}"

# 创建搜索工具实例
search_tool = CustomSearchTool()

# 包装成LangChain Tool
my_search_tool = Tool(
    name="Internet_Search",
    func=search_tool.search,
    description="一个强大的互联网搜索引擎，可以查找最新的在线信息。"
)

# 初始化Agent
llm = Ollama(model="deepseek-r1:7b", temperature=0.3)  # 稍高temperature让搜索更有创意
tools = [my_search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,  # 添加错误处理
    max_iterations=3,  # 限制迭代次数
    early_stopping_method="generate"  # 添加停止条件
)


# 正确的对话处理
def run_agent_query(query: str):
    try:
        response = agent.run({
            "input": query,
            "chat_history": []  # 维持对话历史
        })
        return response
    except Exception as e:
        return f"执行出错: {str(e)}"


if __name__ == "__main__":
    queries = [
        "LangChain是什么？",
        "DeepSeek最新的模型是什么？",
        "2023年最佳AI框架有哪些？"
    ]
    
    for query in queries:
        print(f"\n=== 查询: {query} ===")
        try:
            # response = agent.run(query)
            response = run_agent_query(query)

            print(f"回答: {response}")
        except Exception as e:
            print(f"执行出错: {str(e)}")


"""
PS E:\langchain_learn\llmLearn> & D:/tools/tools/envs/tmp/python.exe e:/langchain_learn/llmLearn/langchain-agent-demo/src/agent_demo2.py
e:\langchain_learn\llmLearn\langchain-agent-demo\src\agent_demo2.py:106: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm = Ollama(model="deepseek-r1:7b", temperature=0.3)  # 稍高temperature让搜索更有创意
e:\langchain_learn\llmLearn\langchain-agent-demo\src\agent_demo2.py:109: LangChainDeprecationWarning: LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph. LangGraph offers a more flexible and full-featured framework for building agents, including support for tool-calling, persistence of state, and human-in-the-loop workflows. For details, refer to the `LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_ as well as guides for `Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_ and LangGraph's `Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_.
  agent = initialize_agent(

=== 查询: LangChain是什么？ ===
e:\langchain_learn\llmLearn\langchain-agent-demo\src\agent_demo2.py:124: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
  response = agent.run({


> Entering new AgentExecutor chain...
<think>
好的，用户问的是“LangChain是什么？”。我需要解释一下LangChain是什么，并且提供一些关键点。

首先，LangChain是一个用于构建复杂AI应用的框架。它帮助开发者组织和管理模型之间的流程，提升效率。

其次，它支持多任务处理，允许一个模型完成多个任务或与多个模型协作。

此外，LangChain提供了模块化设计，便于扩展和定制。

最后，它在自然语言处理、自动化服务等领域有广泛应用。
</think>

LangChain 是一个用于构建复杂 AI 应用的框架。它帮助开发者组织和管理模型之间的流程，提升效率，并支持多任务处理和与多 个模型协作。其模块化设计使其易于扩展和定制，广泛应用于自然语言处理和自动化服务等领域。

Action:
```json
{
  "action": "Final Answer",
  "action_input": "LangChain 是一个用于构建复杂 AI 应用的框架，帮助开发者组织和管理模型之间的流程，提升效率，并支持多任务处理和与多个模型协作。其模块化设计使其易于扩展和定制，广泛应用于自然语言处理和自动化服务等领域。"
}
```

> Finished chain.
回答: LangChain 是一个用于构建复杂 AI 应用的框架，帮助开发者组织和管理模型之间的流程，提升效率，并支持多任务处理和 与多个模型协作。其模块化设计使其易于扩展和定制，广泛应用于自然语言处理和自动化服务等领域。

=== 查询: DeepSeek最新的模型是什么？ ===


> Entering new AgentExecutor chain...
<think>
好，我现在需要回答用户的问题：“DeepSeek最新的模型是什么？”。首先，我应该明确用户想知道的是DeepSeek公司最近发布的新 模型。为了找到准确的信息，我可以使用互联网搜索工具来查找最新的更新。

接下来，我会输入查询语句，比如“DeepSeek latest model 2023”或者类似的关键词，以确保能获取到最新的信息。这样可以帮助 我快速定位到最新的发布动态。

一旦搜索结果出来了，我需要从中提取出模型名称和相关的详细信息，如版本号、发布日期等。如果有多个模型更新，可能需要指 出最新或最热门的一个。

最后，将这些信息整理成一个简洁明了的回答，确保用户能够清楚地了解DeepSeek的最新模型是什么。
</think>

为了找到DeepSeek最新的模型，我使用了Internet Search工具进行查询。

Action:
{
  "action": "Internet_Search",
  "action_input": "DeepSeek latest model"
}



> Finished chain.
回答: <think>
好，我现在需要回答用户的问题：“DeepSeek最新的模型是什么？”。首先，我应该明确用户想知道的是DeepSeek公司最近发布的新 模型。为了找到准确的信息，我可以使用互联网搜索工具来查找最新的更新。

接下来，我会输入查询语句，比如“DeepSeek latest model 2023”或者类似的关键词，以确保能获取到最新的信息。这样可以帮助 我快速定位到最新的发布动态。

一旦搜索结果出来了，我需要从中提取出模型名称和相关的详细信息，如版本号、发布日期等。如果有多个模型更新，可能需要指 出最新或最热门的一个。

最后，将这些信息整理成一个简洁明了的回答，确保用户能够清楚地了解DeepSeek的最新模型是什么。
</think>

为了找到DeepSeek最新的模型，我使用了Internet Search工具进行查询。

Action:
{
  "action": "Internet_Search",
  "action_input": "DeepSeek latest model"
}



=== 查询: 2023年最佳AI框架有哪些？ ===


> Entering new AgentExecutor chain...
<think>
好，我现在需要回答用户的问题：“2023年最佳AI框架有哪些？”首先，我应该明确用户的需求是寻找当前最佳的AI框架，并且他们 特别提到的是“最佳”，所以可能需要最新的信息。

接下来，我会考虑如何获取准确的信息。使用互联网搜索工具来查找最新的排名和评价应该是最直接的方法。这样可以确保得到的 信息是最新的，符合2023年的标准。

在进行搜索时，我应该明确搜索关键词，比如“2023 best AI frameworks”或者类似的表达，以确保结果的相关性和准确性。此外，还需要考虑是否有多个框架被广泛认可，并且用户可能对排名的具体内容感兴趣，如每个框架的优势、适用领域等。

完成搜索后，我会整理出几个主要的AI框架，并简要说明它们的特点或优势，这样用户可以根据需要选择最适合自己的工具。     
</think>

{
  "action": "Internet Search",
  "action_input": "2023年最佳AI框架有哪些"
}

> Finished chain.
回答: <think>
好，我现在需要回答用户的问题：“2023年最佳AI框架有哪些？”首先，我应该明确用户的需求是寻找当前最佳的AI框架，并且他们 特别提到的是“最佳”，所以可能需要最新的信息。

接下来，我会考虑如何获取准确的信息。使用互联网搜索工具来查找最新的排名和评价应该是最直接的方法。这样可以确保得到的 信息是最新的，符合2023年的标准。

在进行搜索时，我应该明确搜索关键词，比如“2023 best AI frameworks”或者类似的表达，以确保结果的相关性和准确性。此外，还需要考虑是否有多个框架被广泛认可，并且用户可能对排名的具体内容感兴趣，如每个框架的优势、适用领域等。

完成搜索后，我会整理出几个主要的AI框架，并简要说明它们的特点或优势，这样用户可以根据需要选择最适合自己的工具。     
</think>

{
  "action": "Internet Search",
  "action_input": "2023年最佳AI框架有哪些"
}
PS E:\langchain_learn\llmLearn> 
"""
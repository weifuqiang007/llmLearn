from langchain_community.llms.ollama import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
import re
# from duckduckgo_search import ddg
from typing import List, Dict
import json
from duckduckgo_search import DDGS

class RobustSearchTool:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def _clean_text(self, text: str) -> str:
        """清除HTML标签和多余空白"""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _fetch_page_content(self, url: str) -> str:
        """获取网页内容"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除不需要的元素
            for element in soup(['script', 'style', 'iframe', 'noscript']):
                element.decompose()
            
            return self._clean_text(soup.get_text())
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    def _format_results(self, results: List[Dict]) -> str:
        """格式化搜索结果"""
        output = []
        for i, result in enumerate(results[:self.max_results], 1):
            content_preview = result.get('content', '')[:300] + "..." if result.get('content') else "无内容预览"
            output.append(
                f"【结果 {i}】\n"
                f"标题: {result.get('title', '无标题')}\n"
                f"链接: {result.get('url', result.get('link', '无链接'))}\n"
                f"摘要: {result.get('snippet', result.get('body', '无摘要'))}\n"
                f"内容预览: {content_preview}\n"
            )
        return "\n".join(output) if output else "未找到相关信息"

    def search(self, query: str) -> str:
        """执行真实网络搜索"""
        try:
            # 1. 使用DuckDuckGo获取搜索结果
            # 1. 使用DuckDuckGo获取搜索结果
            ddg_results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    ddg_results.append(r)
            if not ddg_results:
                return "未找到相关搜索结果"
            
            # 2. 获取页面内容
            enriched_results = []
            for result in ddg_results:
                if len(enriched_results) >= self.max_results:
                    break
                
                url = result.get('href', '') or result.get('link', '')
                if url:
                    content = self._fetch_page_content(url)
                    result['content'] = content
                    enriched_results.append(result)
            
            # 3. 格式化输出
            return self._format_results(enriched_results)
            
        except Exception as e:
            return f"搜索出错: {str(e)}"

# 初始化搜索工具
search_tool = Tool(
    name="InternetSearch",
    func=RobustSearchTool().search,
    description="可靠的互联网搜索引擎，支持获取网页内容",
)

# 初始化LLM
llm = Ollama(model="deepseek-r1:7b", temperature=0.3)

# 创建Agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

def run_query(query: str) -> str:
    """执行查询并处理结果"""
    try:
        response = agent.invoke({"input": query})
        return response['output']
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
        print(run_query(query))

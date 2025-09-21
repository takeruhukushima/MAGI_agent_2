from langchain_community.tools.tavily_search import TavilySearchResults

# 1. まず、個別のツール（Tavily検索）を作成します。
#    これが search_executor_chain が探しているものです。
tavily_tool = TavilySearchResults(max_results=3)

# 2. 次に、この個別のツールを使って、将来的に他のツールも追加できる
#    「道具一式セット」のリストを作成します。
tools = [tavily_tool]

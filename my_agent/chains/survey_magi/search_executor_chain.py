from typing import List, Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command  # Commandをインポート

class SearchExecutorChain:
    """
    検索クエリを実行し、ウェブ検索結果を取得するChain。
    Commandを返して次のノードを指示します。
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        LLMと検索ツールを初期化します。
        TAVILY_API_KEYが環境変数に設定されている必要があります。
        """
        self.llm = llm
        self.search_tool = TavilySearchResults(max_results=5)
    
    def __call__(self, state: dict) -> Command[Literal["relevance_filtering"]]:
        """
        Stateからクエリを取得し、検索を実行してCommandを返します。
        """
        print("--- [Chain] Survey MAGI: 3. Executing Search ---")
        
        queries = state.get("search_queries", [])
        if not queries:
            print("  > No queries to execute.")
            # クエリがない場合も、次のノードに進むCommandを返す
            return Command(
                goto="relevance_filtering",
                update={"search_results": []}
            )
        
        all_results = self.run(queries)
        
        print(f"  > Found {len(all_results)} total results.")
        
        # あなたの指定通り、gotoとupdateを持つCommandを返す
        return Command(
            goto="relevance_filtering",
            update={"search_results": all_results}
        )
    
    def run(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        各クエリに対して検索ツールを実行し、結果をリストにまとめます。
        """
        all_results = []
        for query in queries:
            print(f"  > Executing query: {query[:100]}...")
            try :
                results = self.search_tool.invoke({"query": query})
                all_results.extend(results)
            except Exception as e:
                print(f"  > Error executing query '{query}': {e}")
        return all_results
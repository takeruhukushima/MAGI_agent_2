from typing import List, Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.prompts import PromptManager


class QueryList(BaseModel):
    """A list of search queries."""
    queries: List[str] = Field(
        description="A list of 3 to 5 diverse and specific search queries for academic databases.",
        min_items=1,
        max_items=5
    )


class QueryGeneratorChain:
    """
    A chain that generates search queries based on a research topic.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('survey', 'QUERY_GENERATOR')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            QueryList,
            method="function_calling"
        )

    def __call__(self, state: dict) -> Command[Literal["search_executor"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Survey MAGI: 2. Generating Search Queries ---")
        
        # 複数の可能性のあるキーからトピックを取得
        topic = (
            state.get("clarified_theme") or 
            state.get("research_theme") or 
            state.get("topic")
        )
        
        # メッセージからトピックを取得（もしあれば）
        if not topic and "messages" in state and state["messages"]:
            # 最後のユーザーメッセージを取得
            user_messages = [
                msg for msg in state["messages"] 
                if (hasattr(msg, 'type') and msg.type == 'human') or 
                   (isinstance(msg, dict) and msg.get('role') == 'user')
            ]
            
            if user_messages:
                last_user_message = user_messages[-1]
                if hasattr(last_user_message, 'content'):
                    topic = last_user_message.content
                elif isinstance(last_user_message, dict) and 'content' in last_user_message:
                    topic = last_user_message['content']
        
        # トピックが空の場合はエラー
        if not topic:
            raise ValueError("No topic provided for query generation")
            
        # トピックがデフォルトメッセージの場合はエラー
        if topic == "調査テーマが指定されていません。詳細を教えてください。":
            raise ValueError("No valid topic provided for query generation")
            
        print(f"  > Generating queries for topic: {topic}")
        
        try:
            # Generate search queries
            queries = self.run(topic, state.get("messages", []))
            print(f"  > Generated Queries: {queries.queries}")
            
            return Command(
                goto="search_executor",
                update={
                    "search_queries": queries.queries,
                    "research_theme": topic  # トピックを確実に保持
                }
            )
        except Exception as e:
            print(f"  > Error generating queries: {str(e)}")
            # フォールバックとしてトピック自体をクエリとして使用
            return Command(
                goto="search_executor",
                update={
                    "search_queries": [topic],
                    "research_theme": topic
                }
            )

    def run(self, topic: str, messages: list[BaseMessage] = None) -> QueryList:
        """Generate search queries for the given topic."""
        try:
            return self.chain.invoke({
                "topic": topic,
                "conversation_history": self._format_messages(messages) if messages else ""
            })
        except Exception as e:
            # Fallback to using the topic as a single query
            return QueryList(queries=[topic])
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for the prompt."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# Create a single instance of the chain
from my_agent.settings import settings

query_generator_chain = QueryGeneratorChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

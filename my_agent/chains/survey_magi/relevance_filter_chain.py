from typing import List, Literal, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.prompts import PromptManager


class RelevantDocument(BaseModel):
    """A single relevant document with its summary."""
    url: str = Field(..., description="The URL of the relevant document")
    title: str = Field(..., description="The title of the document")
    summary: str = Field(..., description="A concise summary of the document's relevance")
    relevance_score: float = Field(..., description="A score from 0 to 1 indicating relevance")


class RelevanceScores(BaseModel):
    """A list of documents with relevance scores."""
    relevant_documents: List[RelevantDocument] = Field(
        ...,
        description="List of documents with their relevance scores"
    )


class RelevanceFilterChain:
    """
    A chain that filters search results based on relevance to the research topic.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('survey', 'RELEVANCE_FILTER')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            RelevanceScores,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Command[Literal["document_summarizer"]]:
        """Execute the chain and return a Command to proceed to the next node."""
        print("--- [Chain] Survey MAGI: 4. Filtering Relevance ---")
        
        # デバッグ: 入力状態を確認
        print(f"  > DEBUG - Input state keys: {list(state.keys())}")
        
        topic = state.get("research_theme", "N/A")
        search_results = state.get("search_results", [])
        
        print(f"  > DEBUG - Topic: {topic}")
        print(f"  > DEBUG - Search results count: {len(search_results)}")
        
        if not search_results:
            print("  > No search results to filter.")
            return Command(
                goto="document_summarizer",
                update={"relevant_documents": []}
            )
        
        try:
            filtered = self.run(topic, search_results)
            relevant_docs = [doc.dict() for doc in filtered.relevant_documents]
            print(f"  > Filtered down to {len(relevant_docs)} relevant documents.")
            
            # デバッグ: 出力を確認
            print(f"  > DEBUG - Output relevant_docs count: {len(relevant_docs)}")
            if relevant_docs:
                print(f"  > DEBUG - First document keys: {list(relevant_docs[0].keys())}")
            
            # 正常にフィルタリングされたドキュメントを次のステップに渡す
            return Command(
                goto="document_summarizer",
                update={"relevant_documents": relevant_docs}
            )
            
        except Exception as e:
            print(f"  > Error during relevance filtering: {e}")
            # エラーが発生した場合も、空のリストを渡して次のステップに進む
            return Command(
                goto="document_summarizer",
                update={"relevant_documents": []}
            )
    
    def run(self, topic: str, documents: List[Dict[str, Any]]) -> RelevanceScores:
        """Filter documents based on relevance to the topic."""
        try:
            print(f"  > DEBUG run() - Invoking LLM with topic: {topic}")
            print(f"  > DEBUG run() - Document count: {len(documents)}")
            
            result = self.chain.invoke({
                "research_theme": topic,
                "search_results": documents
            })
            
            print(f"  > DEBUG run() - LLM returned {len(result.relevant_documents)} documents")
            return result
            
        except Exception as e:
            print(f"  > DEBUG run() - Exception: {e}")
            raise RuntimeError(f"Relevance filtering failed: {str(e)}")


# Create a single instance of the chain
from my_agent.settings import settings

relevance_filter_chain = RelevanceFilterChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)
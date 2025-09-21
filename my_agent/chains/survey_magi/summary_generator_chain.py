from typing import List, Literal, Dict, Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from my_agent.prompts import PromptManager


class SurveySummary(BaseModel):
    """The final survey summary with key findings and references."""
    overview: str = Field(..., description="A high-level overview of the research findings")
    key_points: List[str] = Field(..., description="List of key findings from the research")
    references: List[Dict[str, str]] = Field(..., description="List of references with titles and URLs")


class SummaryGeneratorChain:
    """
    A chain that generates a comprehensive summary from relevant documents.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('survey', 'SUMMARY_GENERATOR')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            SurveySummary,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Dict[str, Any]:
        print("--- [Chain] Survey MAGI: 5. Generating Summary ---")
        # state デバッグ
        print(f"  > DEBUG - Full state keys: {list(state.keys())}")
        print(f"  > DEBUG - Full state: {state}")

        topic = state.get("research_theme", "N/A")
        relevant_documents = state.get("relevant_documents", [])

        print(f"  > Debug - Topic: {topic}")
        print(f"  > Debug - Number of relevant documents: {len(relevant_documents) if relevant_documents else 0}")
        print(f"  > Debug - Type of relevant_documents: {type(relevant_documents)}")
        print(f"  > Debug - relevant_documents content: {relevant_documents}")

        # 関連情報なし
        if not relevant_documents or len(relevant_documents) == 0:
            message = "調査を完了しましたが、関連する情報が見つかりませんでした。"
            interrupt(message)

        # 有効な文書か確認
        valid_documents = [doc for doc in relevant_documents if doc and isinstance(doc, dict)]
        if not valid_documents:
            message = "調査を完了しましたが、有効な文書が見つかりませんでした。"
            interrupt(message)

        print(f"  > Processing {len(valid_documents)} valid documents...")
        summary = self.run(topic, valid_documents)
        print("  > Survey summary generated successfully.")

        overview_message = f"調査を完了しました。{topic}に関する要約:\n\n{summary.overview}"
        interrupt(overview_message)
        # interruptがraiseされ、これ以降には到達しません
        # 有効な文書か確認
        valid_documents = [doc for doc in relevant_documents if doc and isinstance(doc, dict)]
        if not valid_documents:
            message = "調査を完了しましたが、有効な文書が見つかりませんでした。"
            interrupt(message)

        print(f"  > Processing {len(valid_documents)} valid documents...")
        summary = self.run(topic, valid_documents)
        print("  > Survey summary generated successfully.")

        overview_message = f"調査を完了しました。{topic}に関する要約:\n\n{summary.overview}"
        interrupt(overview_message)
        # interruptがraiseされ、これ以降には到達しません

    def run(self, topic: str, documents: List[Dict[str, Any]]) -> SurveySummary:
        """Generate a summary from the relevant documents."""
        try:
            # Additional validation before invoking the chain
            if not documents:
                raise ValueError("No documents provided for summary generation")
            
            # Ensure documents have required fields
            processed_documents = []
            for doc in documents:
                if isinstance(doc, dict):
                    processed_documents.append(doc)
                else:
                    print(f"  > Warning: Skipping invalid document: {type(doc)}")
            
            if not processed_documents:
                raise ValueError("No valid documents found after processing")
            
            # デバッグ：LLM呼び出し前
            print(f"  > DEBUG run() - About to call LLM with:")
            print(f"    - research_theme: {topic}")
            print(f"    - relevant_docs count: {len(processed_documents)}")
            
            # 本番用: structured outputを使用
            result = self.chain.invoke({
                "research_theme": topic,         # "research_theme"から"topic"に変更
                "relevant_docs": processed_documents # "relevant_docs"から"documents"に変更
            })
            
            print(f"  > DEBUG run() - Structured LLM response received successfully")
            return result
            
        except Exception as e:
            print(f"  > DEBUG run() - Exception in run(): {e}")
            raise RuntimeError(f"Summary generation failed: {str(e)}")


# Create a single instance of the chain
from my_agent.settings import settings

summary_generator_chain = SummaryGeneratorChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)
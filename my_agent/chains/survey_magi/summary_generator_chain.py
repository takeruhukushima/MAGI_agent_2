import asyncio
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
        prompt_template = PromptManager.get_prompt('survey', 'SUMMARY_GENERATOR')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            SurveySummary,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Dict[str, Any]:
        print("--- [Chain] Survey MAGI: 5. Generating Summary ---")
        topic = state.get("research_theme", "N/A")
        relevant_documents = state.get("relevant_documents", [])
        
        # ▼▼▼ 修正点①: stateからindividual_summariesを取得する ▼▼▼
        individual_summaries = state.get("individual_summaries", [])

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
        
        # ▼▼▼ 修正点②: runメソッドにindividual_summariesを渡す ▼▼▼
        summary = self.run(topic, valid_documents, individual_summaries)
        print("  > Survey summary generated successfully.")

        # 1. すべての情報を整形して1つのメッセージを作成する
        final_message_parts = []
        final_message_parts.append(f"調査を完了しました。**{topic}**に関する要約です。\n")
        final_message_parts.append(f"### 概要\n{summary.overview}\n")
        
        if summary.key_points:
            final_message_parts.append("### キーポイント")
            for i, point in enumerate(summary.key_points, 1):
                final_message_parts.append(f"{i}. {point}")
            final_message_parts.append("") # 改行のため
            
        if summary.references:
            final_message_parts.append("### 参考文献")
            for i, ref in enumerate(summary.references, 1):
                if isinstance(ref, dict):
                    title = ref.get('title', 'No Title')
                    url = ref.get('url', '#')
                    final_message_parts.append(f"{i}. {title}: {url}")
                else:
                    final_message_parts.append(f"{i}. {ref}")
        
        final_message = "\n".join(final_message_parts)
        interrupt(final_message)


    def run(self, topic: str, documents: List[Dict[str, Any]], individual_summaries: List[Dict[str, Any]]) -> SurveySummary:
        """Generate a summary from the relevant documents."""
        try:
            if not documents:
                raise ValueError("No documents provided for summary generation")
            
            # ... (processed_documentsの準備までは同じ)
            processed_documents = [doc for doc in documents if isinstance(doc, dict)]
            if not processed_documents:
                raise ValueError("No valid documents found after processing")

            # ▼▼▼ 修正点: LLMに渡す前に content を削除する ▼▼▼
            # 参考文献リスト用に、content以外の情報だけを抽出
            docs_for_prompt = [
                {k: v for k, v in doc.items() if k != 'content'} 
                for doc in processed_documents
            ]
            # --- ▼▼▼ デバッグコードを追加 ▼▼▼ ---
            print("--- DEBUG: Checking summary lengths ---")
            total_chars = sum(len(s.get('summary', '')) for s in individual_summaries)
            print(f"  > Total characters in all individual_summaries: {total_chars}")
            if individual_summaries:
                print(f"  > Example summary length: {len(individual_summaries[0].get('summary', ''))} chars")
            print("------------------------------------")
            # --- ▲▲▲ デバッグコードここまで ▲▲▲ ---
            
            print(f"  > DEBUG run() - About to call LLM with:")
            print(f"    - research_theme: {topic}")
            # docs_for_prompt を渡すように変更
            print(f"    - relevant_docs (metadata only) count: {len(docs_for_prompt)}")
            print(f"    - individual_summaries count: {len(individual_summaries)}")

            result = self.chain.invoke({
                "research_theme": topic,
                # ▼▼▼ 修正点: contentを削除したリストを渡す ▼▼▼
                "relevant_docs": docs_for_prompt,
                "individual_summaries": individual_summaries
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
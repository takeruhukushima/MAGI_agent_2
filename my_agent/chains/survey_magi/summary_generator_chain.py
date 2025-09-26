import json
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.prompts import PromptManager
from my_agent.settings import settings


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
        individual_summaries = state.get("individual_summaries", [])

        if not relevant_documents or len(relevant_documents) == 0:
            message = "調査を完了しましたが、関連する情報が見つかりませんでした。"
            interrupt(message)
            return {"survey_summary": None, "research_theme": topic,"status": "summary_failed"}


        valid_documents = [doc for doc in relevant_documents if doc and isinstance(doc, dict)]
        if not valid_documents:
            message = "調査を完了しましたが、有効な文書が見つかりませんでした。"
            interrupt(message)
            return {"survey_summary": {}, "research_theme": topic}

        print(f"  > Processing {len(valid_documents)} valid documents...")
        
        summary = self.run(topic, valid_documents, individual_summaries)
        print("  > Survey summary generated successfully.")

        # ユーザーへの最終メッセージを作成して表示
        final_message_parts = []
        final_message_parts.append(f"調査を完了しました。**{topic}**に関する要約です。\n")
        final_message_parts.append(f"### 概要\n{summary.overview}\n")
        if summary.key_points:
            final_message_parts.append("### キーポイント")
            for i, point in enumerate(summary.key_points, 1):
                final_message_parts.append(f"{i}. {point}")
            final_message_parts.append("")
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

        # Stateを更新するための辞書を返す
        return {
            "survey_summary": summary.dict(),
            "research_theme": topic,
            "status": "success",
            "message": final_message
        }

    def run(self, topic: str, documents: List[Dict[str, Any]], individual_summaries: List[Dict[str, Any]]) -> SurveySummary:
        try:
            # ... (runメソッドのロジックは変更なし) ...
            docs_for_prompt = [
                {k: v for k, v in doc.items() if k != 'content'} 
                for doc in documents
            ]
            
            result = self.chain.invoke({
                "research_theme": topic,
                "relevant_docs": docs_for_prompt,
                "individual_summaries": individual_summaries
            })
            
            # 参考文献情報を手動で設定
            if hasattr(result, 'references'):
                docs_for_references = [
                    {'title': doc.get('title', 'No Title'), 'url': doc.get('url', '#')}
                    for doc in documents
                ]
                result.references = docs_for_references
            
            return result
        except Exception as e:
            raise RuntimeError(f"Summary generation failed: {str(e)}")
from typing import List, Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.types import Command  # Commandをインポート

class DocumentSummary(BaseModel):
    summary: str = Field(description="A concise summary of the key points from the document.")

class DocumentSummarizerChain:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            "Summarize the key findings from the following document based on the research theme '{research_theme}'.\n\nDocument Content:\n{document_content}"
        )
        self.chain = self.prompt | self.llm.with_structured_output(DocumentSummary)

    def __call__(self, state: dict) -> Command[Literal["summary_generator"]]:
        """Execute the chain and return a Command to proceed."""
        print("--- [Chain] Survey MAGI: 5a. Summarizing Individual Documents (Map) ---")
        theme = state.get("research_theme")
        documents = state.get("relevant_documents", [])
        
        summaries = []
        for doc in documents:
            try:
                print(f"  > Summarizing doc: {doc.get('title')}")
                result = self.chain.invoke({
                    "research_theme": theme,
                    "document_content": doc.get("content")
                })
                summaries.append({"title": doc.get("title"), "url": doc.get("url"), "summary": result.summary})
            except Exception as e:
                print(f"  > Failed to summarize a document: {e}")
        
        # 修正点：dictの代わりにCommandオブジェクトを返す
        return Command(
            goto="summary_generator",  # 次に進むノード名を指定
            update={"individual_summaries": summaries}  # Stateを更新するデータを指定
        )
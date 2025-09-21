from typing import Dict, Any, Literal, List, Optional
import re

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.settings import settings
from my_agent.prompts import PromptManager


class ReviewComment(BaseModel):
    """A single comment or suggestion from the review."""
    issue_type: str = Field(..., description="Type of issue (e.g., 'typo', 'logic', 'formatting')")
    location: str = Field(..., description="Location of the issue (section, line number, etc.)")
    current_text: str = Field(..., description="The current text that has issues")
    suggested_change: str = Field(..., description="Suggested improvement or correction")
    explanation: str = Field(..., description="Explanation of why this change is suggested")
    severity: str = Field("medium", description="Severity of the issue (low, medium, high)")


class DocumentReview(BaseModel):
    """The complete review of the document."""
    overall_quality: int = Field(..., description="Overall quality score (1-10)")
    summary: str = Field(..., description="Summary of the review")
    comments: List[ReviewComment] = Field(default_factory=list, description="List of specific issues found")
    revised_content: str = Field(..., description="The revised content with suggested changes")


class FinalReviewChain:
    """
    A chain that performs a final review of the generated document.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt = PromptManager.get_prompt("report", "FINAL_REVIEWER")
        self.prompt = ChatPromptTemplate.from_template(prompt)
        self.chain = self.prompt | self.llm.with_structured_output(
            DocumentReview,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Command[Literal["tex_generator"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Report MAGI: 6. Final Review ---")
        
        latex_document = state.get("latex_document", {})
        paper_structure = state.get("paper_structure", {})
        
        try:
            review = self.run(
                content=latex_document.get("content", ""),
                title=paper_structure.get("title", "Research Paper")
            )
            
            print(f"  > Review complete. Document quality: {review.overall_quality}/10")
            
            return Command(
                goto="tex_generator",
                update={
                    "document_review": review.dict(),
                    "final_document": review.revised_content,
                    "status": "review_completed"
                }
            )
            
        except Exception as e:
            print(f"  > Error during final review: {e}")
            return Command(
                goto="tex_generator",
                update={
                    "status": "review_failed",
                    "error": str(e)
                }
            )
    
    def run(self, content: str, title: str = "") -> DocumentReview:
        """Review the document and provide feedback."""
        try:
            # Clean the content by removing LaTeX comments and extra whitespace
            cleaned_content = self._clean_latex(content)
            
            # Get the review from the LLM
            review = self.chain.invoke({
                "document_title": title,
                "document_content": cleaned_content
            })
            
            # If the revised content is not provided, use the original content
            if not review.revised_content or review.revised_content.strip() == "":
                review.revised_content = content
                
            return review
            
        except Exception as e:
            raise RuntimeError(f"Document review failed: {str(e)}")
    
    def _clean_latex(self, content: str) -> str:
        """Clean LaTeX content by removing comments and normalizing whitespace."""
        if not content:
            return ""
            
        # Remove LaTeX comments
        content = re.sub(r'(?<!\\)%.*$', '', content, flags=re.MULTILINE)
        
        # Normalize whitespace
        content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
        
        return content


# Create a single instance of the chain
final_review_chain = FinalReviewChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

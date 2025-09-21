from typing import Dict, Any, Literal, Optional
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field


class AggregatedContent(BaseModel):
    """Structured representation of all content needed for report generation."""
    research_theme: str = Field(..., description="The main research theme")
    research_goal: str = Field(..., description="The main research goal or objective")
    survey_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of the literature survey"
    )
    methodology: Dict[str, Any] = Field(
        default_factory=dict,
        description="Research methodology details"
    )
    experimental_design: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details about the experimental design"
    )
    execution_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from code execution or simulations"
    )
    analysis_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from data analysis"
    )
    references: Dict[str, str] = Field(
        default_factory=dict,
        description="References and citations used in the research"
    )


class ContentAggregatorChain:
    """
    A chain that aggregates content from various sources into a structured format.
    This chain doesn't require an LLM as it only performs data transformation.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize with an optional LLM instance."""
        self.llm = llm  # Not used in this chain, but kept for consistency
    
    def __call__(self, state: dict) -> Command[Literal["section_writer"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Report MAGI: 3. Aggregating Content ---")
        
        try:
            # Extract and structure content from the state
            aggregated = self.run(state)
            
            print("  > Content aggregation completed successfully.")
            
            return Command(
                goto="section_writer",
                update={
                    "aggregated_content": aggregated.dict(),
                    "status": "content_aggregated"
                }
            )
            
        except Exception as e:
            print(f"  > Error during content aggregation: {e}")
            return Command(
                goto="section_writer",  # Still try to proceed to writing with what we have
                update={
                    "aggregated_content": {"error": str(e)},
                    "status": "content_aggregation_error"
                }
            )
    
    def run(self, state: dict) -> AggregatedContent:
        """Aggregate content from the state into a structured format."""
        try:
            # Safely extract data from the state with defaults
            return AggregatedContent(
                research_theme=state.get("research_theme", ""),
                research_goal=state.get("research_goal", ""),
                survey_summary=state.get("survey_summary", {}),
                methodology=state.get("methodology", {}),
                experimental_design=state.get("experimental_design", {}),
                execution_results=state.get("execution_results", {}),
                analysis_results=state.get("analysis_results", {}),
                references=state.get("references", {})
            )
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate content: {str(e)}")


# Create a single instance of the chain
content_aggregator_chain = ContentAggregatorChain()

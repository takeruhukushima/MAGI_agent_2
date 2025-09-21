from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from my_agent.utils.state import AgentState
from my_agent.prompts import AnalysisPrompts
from my_agent.settings import settings

class AnalysisMethodType(str, Enum):
    """Types of analysis methods that can be selected."""
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    EXPLORATORY = "exploratory"
    CAUSAL = "causal"
    MACHINE_LEARNING = "machine_learning"
    TIME_SERIES = "time_series"
    OTHER = "other"


class AnalysisMethod(BaseModel):
    """A selected analysis method with metadata."""
    name: str = Field(
        ...,
        description="The name of the analysis method."
    )
    method_type: AnalysisMethodType = Field(
        ...,
        description="The type of analysis method."
    )
    description: str = Field(
        ...,
        description="A detailed description of the analysis method."
    )
    rationale: str = Field(
        ...,
        description="The reasoning behind selecting this method for the given data and research goal."
    )
    implementation_notes: str = Field(
        default="",
        description="Notes on how to implement this analysis, including any required preprocessing steps."
    )
    required_libraries: List[str] = Field(
        default_factory=list,
        description="List of Python libraries required to implement this analysis."
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Key assumptions that must be met for this analysis to be valid."
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Potential limitations or constraints of this analysis method."
    )
    references: List[str] = Field(
        default_factory=list,
        description="References or citations for this analysis method."
    )


class MethodSelectionResult(BaseModel):
    """The result of method selection, including primary and alternative methods."""
    primary_method: AnalysisMethod = Field(
        ...,
        description="The primary recommended analysis method."
    )
    alternative_methods: List[AnalysisMethod] = Field(
        default_factory=list,
        description="Alternative analysis methods that could also be appropriate."
    )
    comparison: str = Field(
        default="",
        description="Comparison of the selected method with alternatives."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the selected method (0.0 to 1.0)."
    )


class MethodSelectorChain:
    """
    A chain that selects the most appropriate analysis methods based on the data and research goals.
    
    This chain evaluates the available data, research questions, and context to recommend
    suitable statistical and analytical approaches.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the MethodSelectorChain with an optional LLM."""
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature
        )
        
        # Use the prompt template directly from AnalysisPrompts
        self.prompt_template = ChatPromptTemplate.from_template(
            template=AnalysisPrompts.METHOD_SELECTOR,
            template_format="f-string"
        )
        
        # Create the chain with structured output
        self.chain = self.prompt_template | self.llm.with_structured_output(MethodSelectionResult)
    
    def _get_research_context(self, state: AgentState) -> Dict[str, Any]:
        """Extract relevant context for method selection from the state."""
        research_goal = state.get("research_goal", "Not specified.")
        
        # Get data characteristics from validation results if available
        data_characteristics = {
            "data_type": "unknown",
            "sample_size": 0,
            "variables": [],
            "issues": []
        }
        
        if "data_validation_summary" in state:
            data_characteristics["issues"] = state.get("data_quality_issues", [])
        
        # Get dataset preview
        execution_results = state.get("execution_results", {})
        if isinstance(execution_results, dict):
            if 'output_data' in execution_results and 'dataset' in execution_results['output_data']:
                dataset = execution_results['output_data']['dataset']
                data_characteristics["data_type"] = type(dataset).__name__
                
                # Try to get sample size if it's a list or dict
                if isinstance(dataset, (list, dict)):
                    data_characteristics["sample_size"] = len(dataset)
                
                # Try to infer variables/columns
                if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], dict):
                    data_characteristics["variables"] = list(dataset[0].keys())
                elif isinstance(dataset, dict):
                    data_characteristics["variables"] = list(dataset.keys())
        
        return {
            "research_goal": research_goal,
            "data_characteristics": data_characteristics,
            "data_validation_summary": state.get("data_validation_summary", "No validation performed."),
            "data_quality_issues": state.get("data_quality_issues", [])
        }
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Select the most appropriate analysis methods based on the current state.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Analysis MAGI: 2. Selecting Analysis Method ---")
        
        try:
            # Get research context and data characteristics
            context = self._get_research_context(state)
            
            # If no data is available, return an error
            if not context["data_characteristics"].get("variables"):
                error_msg = "No data available for method selection. Please ensure data has been loaded and validated."
                print(f"  > {error_msg}")
                return Command(
                    update={
                        "error": error_msg,
                        "messages": [{"role": "system", "content": error_msg}]
                    }
                )
            
            # Select the analysis methods
            result = await self.run(context, state.get("messages", []))
            
            # Prepare the method information for the state
            method_info = {
                "primary_method": result.primary_method.model_dump(),
                "alternative_methods": [m.model_dump() for m in result.alternative_methods],
                "method_selection_confidence": result.confidence,
                "selected_method": result.primary_method.name  # For backward compatibility
            }
            
            print(f"  > Selected primary method: {result.primary_method.name} (confidence: {result.confidence:.2f})")
            if result.alternative_methods:
                print(f"  > Alternative methods: {', '.join(m.name for m in result.alternative_methods)}")
            
            return Command(
                update={
                    **method_info,
                    "messages": [{
                        "role": "system",
                        "content": f"Selected analysis method: {result.primary_method.name}. "
                                  f"Confidence: {result.confidence:.2f}"
                    }]
                }
            )
            
        except Exception as e:
            error_msg = f"Error during method selection: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "error": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, context: Dict[str, Any], messages: Optional[List[Dict[str, str]]] = None) -> MethodSelectionResult:
        """
        Select analysis methods based on the given context.
        
        Args:
            context: Dictionary containing research context and data characteristics
            messages: Previous conversation messages for context
            
        Returns:
            MethodSelectionResult: The selected analysis methods
        """
        try:
            # Get the method selection from the LLM
            result = await self.chain.ainvoke({
                "research_goal": context["research_goal"],
                "data_characteristics": context["data_characteristics"],
                "data_validation_summary": context["data_validation_summary"],
                "data_quality_issues": context["data_quality_issues"]
            })
            
            return result
            
        except Exception as e:
            # Fallback to a default method if LLM call fails
            return MethodSelectionResult(
                primary_method=AnalysisMethod(
                    name="Descriptive Statistics",
                    method_type=AnalysisMethodType.DESCRIPTIVE,
                    description="Basic descriptive statistics including mean, median, standard deviation, etc.",
                    rationale="Fallback method selected due to an error in method selection.",
                    implementation_notes="Use pandas.describe() for a quick overview of the data.",
                    required_libraries=["pandas", "numpy"],
                    assumptions=["Data is numerical or can be meaningfully described with statistics"],
                    limitations=["Does not provide inferential or predictive insights"],
                    references=["https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html"]
                ),
                confidence=0.5
            )


# Create a singleton instance
method_selector_chain_instance = MethodSelectorChain()


async def method_selector_chain(state: AgentState) -> Command:
    """
    Entry point for the method selector chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await method_selector_chain_instance(state)

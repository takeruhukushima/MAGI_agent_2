from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator
import json
from enum import Enum
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from my_agent.utils.state import AgentState
from my_agent.prompts import AnalysisPrompts
from my_agent.settings import settings

class InterpretationSection(BaseModel):
    """A section of the interpretation result."""
    title: str = Field(
        ...,
        description="Title of this interpretation section."
    )
    content: str = Field(
        ...,
        description="Detailed interpretation content."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this interpretation (0.0 to 1.0)."
    )
    related_metrics: List[str] = Field(
        default_factory=list,
        description="List of metrics or data points this interpretation is based on."
    )
    visualization_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for visualizations that could enhance understanding."
    )


class ResultInterpretation(BaseModel):
    """Structured interpretation of analysis results."""
    summary: str = Field(
        ...,
        description="A concise summary of the analysis results."
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="List of key findings from the analysis."
    )
    sections: List[InterpretationSection] = Field(
        default_factory=list,
        description="Detailed interpretation sections."
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitations or caveats of the analysis."
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Suggested next steps or actions based on the results."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the interpretation (0.0 to 1.0)."
    )


class ResultInterpreterChain:
    """
    A chain that interprets analysis results (tables, charts, statistics) in natural language.
    
    This chain takes the output of analysis code execution and provides
    meaningful, context-aware interpretations of the results.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the ResultInterpreterChain with an optional LLM."""
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature
        )
        
        # Use the prompt template directly from AnalysisPrompts
        self.prompt_template = ChatPromptTemplate.from_template(
            template=AnalysisPrompts.RESULT_INTERPRETER,
            template_format="f-string"
        )
        
        # Create the chain with structured output
        self.chain = self.prompt_template | self.llm.with_structured_output(ResultInterpretation)
    
    def _prepare_context(self, state: AgentState) -> Dict[str, Any]:
        """Prepare the context for result interpretation from the agent state."""
        # Get analysis results
        analysis_output = state.get("analysis_output", {})
        if not analysis_output or analysis_output == "No analysis output available.":
            # Try to get execution results if analysis_output is not available
            execution_results = state.get("execution_results", {})
            if isinstance(execution_results, list) and len(execution_results) > 0:
                analysis_output = execution_results[-1].get("output_data", {})
        
        # Get research context
        research_goal = state.get("research_goal", "Not specified.")
        analysis_method = state.get("primary_method", {})
        if isinstance(analysis_method, dict):
            analysis_method = analysis_method.get("name", "Not specified.")
        
        # Get data characteristics
        data_characteristics = state.get("data_characteristics", {})
        
        return {
            "research_goal": research_goal,
            "analysis_method": analysis_method,
            "analysis_output": analysis_output,
            "data_characteristics": data_characteristics,
            "code_description": state.get("code_description", ""),
            "expected_outputs": state.get("expected_outputs", [])
        }
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Interpret the analysis results in the current state.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Analysis MAGI: 4. Interpreting Results ---")
        
        try:
            # Prepare context for interpretation
            context = self._prepare_context(state)
            
            # If no analysis output is available, return an error
            if not context["analysis_output"] or context["analysis_output"] == "No analysis output available.":
                error_msg = "No analysis output available for interpretation. Please run the analysis first."
                print(f"  > {error_msg}")
                return Command(
                    update={
                        "error": error_msg,
                        "messages": [{"role": "system", "content": error_msg}]
                    }
                )
            
            # Interpret the results
            result = await self.run(context, state.get("messages", []))
            
            # Prepare the interpretation for the state
            interpretation = {
                "result_interpretation": result.model_dump(),
                "interpretation_summary": result.summary,
                "key_findings": result.key_findings,
                "interpretation_confidence": result.confidence,
                "messages": [{
                    "role": "system",
                    "content": f"Results interpreted with {result.confidence:.1%} confidence. {result.summary[:100]}..."
                }]
            }
            
            print(f"  > Interpretation complete (confidence: {result.confidence:.1%})")
            if result.key_findings:
                print(f"  > Key findings: {len(result.key_findings)} identified")
            
            return Command(update=interpretation)
            
        except Exception as e:
            error_msg = f"Error interpreting results: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "error": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, context: Dict[str, Any], messages: Optional[List[Dict[str, str]]] = None) -> ResultInterpretation:
        """
        Interpret the analysis results based on the given context.
        
        Args:
            context: Dictionary containing analysis context
            messages: Previous conversation messages for context
            
        Returns:
            ResultInterpretation: The interpretation of the analysis results
        """
        try:
            # Format the analysis output for the prompt
            analysis_output = context["analysis_output"]
            if isinstance(analysis_output, (dict, list)):
                analysis_output_str = json.dumps(analysis_output, indent=2)
                # Limit the size of the output in the prompt
                if len(analysis_output_str) > 3000:
                    analysis_output_str = analysis_output_str[:1500] + "..." + analysis_output_str[-1500:]
            else:
                analysis_output_str = str(analysis_output)
            
            # Generate the interpretation
            result = await self.chain.ainvoke({
                "research_goal": context["research_goal"],
                "analysis_method": context["analysis_method"],
                "analysis_output": analysis_output_str,
                "data_characteristics": json.dumps(context["data_characteristics"], indent=2),
                "code_description": context["code_description"],
                "expected_outputs": ", ".join(context["expected_outputs"])
            })
            
            return result
            
        except Exception as e:
            # Fallback to a simple interpretation if generation fails
            return ResultInterpretation(
                summary="The analysis results could not be fully interpreted automatically.",
                key_findings=[
                    "The analysis completed, but the results could not be fully interpreted.",
                    f"Error during interpretation: {str(e)[:200]}"
                ],
                sections=[
                    InterpretationSection(
                        title="Raw Analysis Output",
                        content=f"The analysis produced the following output:\n\n{str(analysis_output)[:1000]}",
                        confidence=0.5,
                        related_metrics=[],
                        visualization_suggestions=[]
                    )
                ],
                limitations=[
                    "Automatic interpretation failed due to an error.",
                    "The raw analysis output is provided for manual review."
                ],
                next_steps=[
                    "Review the raw analysis output manually.",
                    "Consider re-running the analysis with different parameters."
                ],
                confidence=0.0
            )


# Create a singleton instance
result_interpreter_chain_instance = ResultInterpreterChain()


async def result_interpreter_chain(state: AgentState) -> Command:
    """
    Entry point for the result interpreter chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await result_interpreter_chain_instance(state)

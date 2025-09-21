"""
Conclusion Generator Chain for Research Analysis

This module provides functionality to generate comprehensive conclusions and future work
based on research analysis results. It processes interpretation data and produces
structured conclusions with confidence scores and key findings.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
from my_agent.utils.state import AgentState
from my_agent.prompts import AnalysisPrompts, PromptManager
from my_agent.settings import settings

# Set up logging
logger = logging.getLogger(__name__)

class ConclusionSection(BaseModel):
    """A section of the research conclusion with importance rating.
    
    Attributes:
        title: The title of the conclusion section
        content: Detailed content of the section
        importance: Importance level (high/medium/low)
    """
    title: str = Field(
        ...,
        description="Title of this conclusion section (e.g., 'Key Findings', 'Methodological Limitations')",
        min_length=1,
        max_length=100
    )
    content: str = Field(
        ...,
        description="Detailed content of this section, including evidence and reasoning",
        min_length=10
    )
    importance: str = Field(
        default="medium",
        description="Importance level: 'high', 'medium', or 'low'"
    )
    
    @field_validator('importance')
    def validate_importance(cls, v: str) -> str:
        """Validate that importance is one of the allowed values."""
        v = v.lower()
        if v not in ['high', 'medium', 'low']:
            raise ValueError("Importance must be one of: 'high', 'medium', 'low'")
        return v


class ResearchConclusion(BaseModel):
    """Structured representation of research conclusions and future work.
    
    This model captures the complete output of the conclusion generation process,
    including key findings, limitations, and suggested future work.
    """
    summary: str = Field(
        ...,
        description="A concise summary (1-3 paragraphs) of the research conclusions",
        min_length=50
    )
    key_contributions: List[str] = Field(
        default_factory=list,
        description="List of 3-5 key contributions or findings from the research"
    )
    sections: List[ConclusionSection] = Field(
        default_factory=list,
        description="Detailed conclusion sections with supporting evidence"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="List of limitations or constraints of the current research"
    )
    future_work: List[str] = Field(
        default_factory=list,
        description="List of 3-5 suggested directions for future research"
    )
    implications: List[str] = Field(
        default_factory=list,
        description="List of practical or theoretical implications of the research"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the conclusions (0.0 to 1.0)"
    )


class ConclusionGeneratorChain:
    """
    A chain that generates comprehensive conclusions and future work based on research analysis.
    
    This chain synthesizes the research findings, interpretations, and analysis results
    to produce meaningful conclusions and suggest future research directions. It handles
    the entire conclusion generation process, including context preparation, LLM inference,
    and result processing.
    
    Example:
        ```python
        chain = ConclusionGeneratorChain()
        result = await chain.run({
            "research_goal": "Analyze customer churn",
            "interpretation": {"key_findings": ["Churn is correlated with..."]},
            "analysis_method": "logistic_regression"
        })
        ```
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the ConclusionGeneratorChain with an optional LLM.
        
        Args:
            llm: Optional ChatGoogleGenerativeAI instance. If not provided, a default
                 instance will be created with settings from the settings module.
        """
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature,
            max_output_tokens=2048  # Default value since it's not in settings
        )
        
        try:
            # Get the prompt directly from AnalysisPrompts
            self.prompt_template = ChatPromptTemplate.from_template(
                template=AnalysisPrompts.CONCLUSION_GENERATOR,
                template_format="f-string"
            )
            
            # Create the chain with structured output
            self.chain = self.prompt_template | self.llm.with_structured_output(
                ResearchConclusion,
                method="json_schema"
            )
            
            logger.info("ConclusionGeneratorChain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConclusionGeneratorChain: {str(e)}")
            raise
    
    def _prepare_context(self, state: AgentState) -> Dict[str, Any]:
        """Prepare the context for conclusion generation from the agent state.
        
        Args:
            state: The current agent state containing research data and analysis results
            
        Returns:
            Dict containing structured context for conclusion generation
            
        Raises:
            ValueError: If required context data is missing or invalid
        """
        try:
            # Get research context with validation
            research_goal = state.get("research_goal")
            if not research_goal:
                logger.warning("No research_goal found in state")
                research_goal = "Research goal not specified"
            
            # Get and validate interpretation results
            interpretation = state.get("result_interpretation", {})
            if isinstance(interpretation, str):
                interpretation = {"summary": interpretation}
            elif not isinstance(interpretation, dict):
                logger.warning("Unexpected interpretation format, converting to dict")
                interpretation = {"summary": str(interpretation)}
            
            # Get analysis method with fallback
            analysis_method = state.get("primary_method", {})
            if isinstance(analysis_method, dict):
                analysis_method = analysis_method.get("name", "Not specified")
            elif not isinstance(analysis_method, str):
                analysis_method = str(analysis_method)
            
            # Get execution results (most recent if it's a list)
            execution_results = state.get("execution_results", {})
            if isinstance(execution_results, list):
                execution_results = execution_results[-1] if execution_results else {}
            
            # Get additional context with defaults
            key_findings = state.get("key_findings", [])
            if not isinstance(key_findings, list):
                logger.warning("key_findings is not a list, converting")
                key_findings = [str(key_findings)]
            
            data_characteristics = state.get("data_characteristics", {})
            if not isinstance(data_characteristics, dict):
                logger.warning("data_characteristics is not a dict, using empty dict")
                data_characteristics = {}
            
            return {
                "research_goal": research_goal,
                "interpretation": interpretation,
                "analysis_method": analysis_method,
                "execution_results": execution_results,
                "key_findings": key_findings,
                "data_characteristics": data_characteristics,
                "timestamp": state.get("timestamp"),
                "analysis_id": state.get("analysis_id")
            }
            
        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            raise ValueError(f"Failed to prepare context: {str(e)}")
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Generate research conclusions based on the current state.
        
        This is the main entry point when used in a LangGraph workflow.
        It prepares the context, generates conclusions, and returns a Command
        with the results or an error message.
        
        Args:
            state: The current agent state containing research data
            
        Returns:
            Command: A LangGraph Command with the conclusion data or error
        """
        logger.info("--- [Chain] Analysis MAGI: 5. Generating Conclusion ---")
        
        try:
            # Prepare context for conclusion generation
            context = self._prepare_context(state)
            
            # Validate that we have the minimum required information
            if not context.get("interpretation") or not context["interpretation"].get("summary"):
                error_msg = (
                    "Insufficient interpretation data available for generating conclusions. "
                    "Please ensure the result interpreter has been run successfully."
                )
                logger.warning(error_msg)
                return Command(
                    update={
                        "error": error_msg,
                        "status": "error",
                        "messages": [{"role": "system", "content": error_msg}]
                    }
                )
            
            # Generate the conclusion
            logger.debug("Generating research conclusions...")
            result = await self.run(context, state.get("messages", []))
            
            # Prepare the conclusion information for the state
            conclusion_info = {
                "research_conclusion": result.model_dump(),
                "conclusion_summary": result.summary,
                "key_contributions": result.key_contributions,
                "conclusion_confidence": result.confidence,
                "status": "completed",
                "messages": [{
                    "role": "system",
                    "content": (
                        f"Successfully generated research conclusions with {result.confidence:.1%} confidence. "
                        f"Summary: {result.summary[:150]}..."
                    )
                }],
                "metadata": {
                    "sections_generated": len(result.sections),
                    "key_contributions_count": len(result.key_contributions),
                    "future_work_items": len(result.future_work),
                    "limitations_identified": len(result.limitations)
                }
            }
            
            # Log success with key metrics
            logger.info(
                f"Conclusion generated successfully (confidence: {result.confidence:.1%}). "
                f"Key contributions: {len(result.key_contributions)}, "
                f"Future work items: {len(result.future_work)}"
            )
            
            if result.confidence < 0.5:
                logger.warning(
                    f"Low confidence in generated conclusions ({result.confidence:.1%}). "
                    "Consider reviewing the results carefully."
                )
            
            return Command(update=conclusion_info)
            
        except Exception as e:
            error_msg = f"Error generating conclusions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Create a fallback conclusion with error information
            fallback_conclusion = ResearchConclusion(
                summary=(
                    "An error occurred while generating conclusions. "
                    "The analysis results may be incomplete or invalid."
                ),
                key_contributions=[
                    "The analysis workflow completed but encountered an error during conclusion generation.",
                    "Please review the error details and consider re-running the analysis."
                ],
                limitations=[
                    f"Conclusion generation failed with error: {str(e)[:200]}",
                    "The validity of these conclusions cannot be guaranteed."
                ],
                confidence=0.1
            )
            
            return Command(
                update={
                    "error": error_msg,
                    "status": "error",
                    "research_conclusion": fallback_conclusion.model_dump(),
                    "conclusion_summary": fallback_conclusion.summary,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, context: Dict[str, Any], messages: Optional[List[Dict[str, str]]] = None) -> ResearchConclusion:
        """
        Generate research conclusions based on the given context.
        
        This is the core method that handles the LLM inference and result processing.
        It can be called directly with a context dictionary when not using LangGraph.
        
        Args:
            context: Dictionary containing research context including:
                - research_goal: The original research objective
                - interpretation: Analysis interpretation results
                - analysis_method: The primary analysis method used
                - key_findings: List of key findings from the analysis
                - data_characteristics: Characteristics of the analyzed data
            messages: Optional list of previous conversation messages for context
            
        Returns:
            ResearchConclusion: The generated research conclusions
            
        Raises:
            ValueError: If required context is missing or invalid
            RuntimeError: If the LLM fails to generate valid conclusions
        """
        logger.debug("Starting conclusion generation...")
        
        try:
            # Validate required context
            required_fields = ["research_goal", "interpretation", "analysis_method"]
            for field in required_fields:
                if not context.get(field):
                    raise ValueError(f"Missing required context field: {field}")
            
            # Format the interpretation for the prompt
            interpretation = context["interpretation"]
            if isinstance(interpretation, dict):
                try:
                    interpretation_str = json.dumps(interpretation, indent=2, ensure_ascii=False)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize interpretation: {str(e)}")
                    interpretation_str = str(interpretation)
            else:
                interpretation_str = str(interpretation)
            
            # Prepare key findings with fallback
            key_findings = context.get("key_findings", [])
            if not key_findings:
                key_findings = ["No specific key findings were identified."]
            
            # Prepare data characteristics with fallback
            data_characteristics = context.get("data_characteristics", {})
            if not data_characteristics:
                data_characteristics = {"note": "No specific data characteristics were provided."}
            
            # Generate the conclusion using the LLM
            logger.debug("Invoking LLM for conclusion generation...")
            try:
                result = await self.chain.ainvoke({
                    "research_goal": context["research_goal"],
                    "interpretation": interpretation_str,
                    "analysis_method": context["analysis_method"],
                    "key_findings": "\n- " + "\n- ".join(key_findings),
                    "data_characteristics": json.dumps(data_characteristics, indent=2, ensure_ascii=False),
                    "timestamp": context.get("timestamp", "Not specified"),
                    "analysis_id": context.get("analysis_id", "Not specified")
                })
                
                # Validate the result
                if not result or not result.summary:
                    raise RuntimeError("LLM returned an empty or invalid conclusion")
                
                logger.info(f"Successfully generated conclusion with {len(result.sections)} sections")
                return result
                
            except Exception as llm_error:
                logger.error(f"LLM invocation failed: {str(llm_error)}", exc_info=True)
                raise RuntimeError(f"Failed to generate conclusions: {str(llm_error)}")
            
        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            
            # Create a detailed error message with context
            error_context = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "context_keys": list(context.keys()) if isinstance(context, dict) else "N/A",
                "has_interpretation": bool(context.get("interpretation")) if isinstance(context, dict) else False,
                "has_analysis_method": bool(context.get("analysis_method")) if isinstance(context, dict) else False
            }
            
            logger.error(f"Error context: {json.dumps(error_context, indent=2)}")
            
            # Return a fallback conclusion with error details
            return ResearchConclusion(
                summary=(
                    "The research analysis was completed, but an error occurred during "
                    f"conclusion generation: {type(e).__name__}"
                ),
                key_contributions=[
                    "The analysis workflow completed but encountered an error during conclusion generation.",
                    "Please review the error details and consider re-running the analysis."
                ],
                sections=[
                    ConclusionSection(
                        title="Error Details",
                        content=(
                            f"An error occurred during conclusion generation: {str(e)}\n\n"
                            "This may be due to issues with the input data, analysis results, "
                            "or limitations in the conclusion generation process."
                        ),
                        importance="high"
                    )
                ],
                limitations=[
                    "The conclusions could not be generated due to an error in the process.",
                    "The validity of any partial results cannot be guaranteed.",
                    f"Error type: {type(e).__name__}"
                ],
                future_work=[
                    "Review and fix the error in the conclusion generation process.",
                    "Manually review the analysis results and interpretation.",
                    "Consider adjusting the analysis parameters and retrying."
                ],
                confidence=0.0
            )


# Create a singleton instance for convenience
conclusion_generator_chain_instance = ConclusionGeneratorChain()


async def conclusion_generator_chain(state: AgentState) -> Command:
    """
    Entry point for the conclusion generator chain in LangGraph workflows.
    
    This function provides a simple interface for using the ConclusionGeneratorChain
    within a LangGraph workflow. It delegates to the singleton instance.
    
    Args:
        state: The current agent state containing research data and analysis results
        
    Returns:
        Command: A LangGraph Command with the conclusion data or error information
        
    Example:
        ```python
        from my_agent.chains.analysis_magi.conclusion_generator_chain import conclusion_generator_chain
        
        # In a LangGraph node
        async def my_node(state):
            return await conclusion_generator_chain(state)
        ```
    """
    try:
        # Log the start of the chain
        logger.info("--- [Chain] Analysis MAGI: 5. Generating Conclusion (via entry point) ---")
        
        # Delegate to the singleton instance
        return await conclusion_generator_chain_instance(state)
        
    except Exception as e:
        # This is a last-resort error handler in case the instance's error handling fails
        error_msg = f"Unhandled error in conclusion_generator_chain: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        
        return Command(
            update={
                "error": error_msg,
                "status": "error",
                "messages": [{"role": "system", "content": error_msg}]
            }
        )
    return await conclusion_generator_chain_instance(state)

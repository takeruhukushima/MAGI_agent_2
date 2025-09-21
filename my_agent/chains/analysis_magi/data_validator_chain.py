from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from my_agent.utils.state import AgentState
from my_agent.prompts import AnalysisPrompts
from my_agent.settings import settings

class DataQualityIssue(BaseModel):
    """Represents a data quality issue found during validation."""
    issue_type: str = Field(..., description="The type of data quality issue.")
    description: str = Field(..., description="Description of the issue.")
    severity: str = Field(..., description="Severity level: 'low', 'medium', or 'high'.")
    affected_columns: List[str] = Field(
        default_factory=list,
        description="List of column names affected by this issue, if applicable."
    )
    suggested_fix: Optional[str] = Field(
        None,
        description="Optional suggestion for how to fix this issue."
    )
    
    @field_validator('severity', mode='before')
    def validate_severity(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError("Severity must be one of: 'low', 'medium', 'high'")
        return v


class DataValidationResult(BaseModel):
    """Results of data validation."""
    is_valid: bool = Field(
        ...,
        description="Whether the data is valid and ready for analysis."
    )
    reason: str = Field(
        ...,
        description="A brief explanation of the validation result."
    )
    issues: List[DataQualityIssue] = Field(
        default_factory=list,
        description="List of data quality issues found during validation."
    )
    summary_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics and metrics about the data quality."
    )
    

class DataValidatorChain:
    """
    A chain that validates and preprocesses raw data for analysis.
    
    This chain checks for data quality issues such as missing values, outliers,
    and inconsistencies, and provides suggestions for data cleaning.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the DataValidatorChain with an optional LLM."""
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature
        )
        
        # Use the prompt template directly from AnalysisPrompts
        self.prompt_template = ChatPromptTemplate.from_template(
            template=AnalysisPrompts.DATA_VALIDATOR,
            template_format="f-string"
        )
        
        # Create the chain with structured output
        self.chain = self.prompt_template | self.llm.with_structured_output(DataValidationResult)
    
    def _extract_data_sample(self, execution_results: Dict[str, Any]) -> str:
        """Extract a sample of data for validation."""
        # Try to get the dataset from different possible locations
        if isinstance(execution_results, dict):
            # Case 1: Directly in the execution results
            if 'dataset' in execution_results:
                data = execution_results['dataset']
            # Case 2: Nested under output_data
            elif 'output_data' in execution_results and 'dataset' in execution_results['output_data']:
                data = execution_results['output_data']['dataset']
            # Case 3: Check for list of execution results
            elif isinstance(execution_results.get('execution_results'), list) and len(execution_results['execution_results']) > 0:
                last_execution = execution_results['execution_results'][-1]
                if isinstance(last_execution, dict) and 'output_data' in last_execution:
                    data = last_execution['output_data'].get('dataset', {})
                else:
                    data = {}
            else:
                data = {}
        else:
            data = {}
        
        # Convert to string representation, limiting the size
        data_str = str(data)
        if len(data_str) > 1000:
            return data_str[:500] + "..." + data_str[-500:]
        return data_str
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Validate the data in the current state.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Analysis MAGI: 1. Validating Data ---")
        
        # Get the data to validate
        execution_results = state.get("execution_results", {})
        data_sample = self._extract_data_sample(execution_results)
        
        if not data_sample or data_sample == "{}":
            error_msg = "No data found for validation."
            print(f"  > {error_msg}")
            return Command(
                update={
                    "is_data_valid": False,
                    "data_validation_summary": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
        
        try:
            # Run the validation
            result = await self.run(data_sample, state.get("messages", []))
            
            if result.is_valid:
                print(f"  > Data validation successful: {result.reason}")
                return Command(
                    update={
                        "is_data_valid": True,
                        "data_validation_summary": result.reason,
                        "data_quality_issues": [issue.model_dump() for issue in result.issues],
                        "data_metrics": result.summary_metrics,
                        "messages": [{"role": "system", "content": f"Data validation successful: {result.reason}"}]
                    }
                )
            else:
                print(f"  > Data validation found issues: {result.reason}")
                return Command(
                    update={
                        "is_data_valid": False,
                        "data_validation_summary": result.reason,
                        "data_quality_issues": [issue.model_dump() for issue in result.issues],
                        "data_metrics": result.summary_metrics,
                        "messages": [{"role": "system", "content": f"Data validation issues found: {result.reason}"}],
                        "requires_attention": True
                    }
                )
                
        except Exception as e:
            error_msg = f"Error during data validation: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "is_data_valid": False,
                    "data_validation_summary": error_msg,
                    "error": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, data_sample: str, messages: Optional[List[Dict[str, str]]] = None) -> DataValidationResult:
        """
        Validate the given data sample.
        
        Args:
            data_sample: A sample of the data to validate
            messages: Previous conversation messages for context
            
        Returns:
            DataValidationResult: The validation results
        """
        # Format the prompt with the data sample
        prompt = self.prompt_template.format(data_sample=data_sample)
        
        try:
            # Get the validation result from the LLM
            validation_result = await self.chain.ainvoke({"data_sample": data_sample})
            return validation_result
            
        except Exception as e:
            # Fallback to a basic validation result if LLM call fails
            return DataValidationResult(
                is_valid=False,
                reason=f"Error during validation: {str(e)}",
                issues=[{
                    "issue_type": "validation_error",
                    "description": f"Failed to validate data: {str(e)}",
                    "severity": "high"
                }]
            )


# Create a singleton instance
data_validator_chain_instance = DataValidatorChain()


async def data_validator_chain(state: AgentState) -> Command:
    """
    Entry point for the data validator chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await data_validator_chain_instance(state)

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from langgraph.types import Command

from my_agent.utils.state import AgentState

class ExecutionRecord(BaseModel):
    """A structured record of an execution's results and logs."""
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="The timestamp when the execution was logged."
    )
    log: str = Field(
        ...,
        description="The execution log or output messages."
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="The structured output data from the execution."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution."
    )
    success: bool = Field(
        True,
        description="Whether the execution was successful."
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the execution failed."
    )


class DataLoggerChain:
    """
    A chain that systematically records execution results and logs to the AgentState.
    
    This chain is responsible for persisting execution outcomes, including
    logs, output data, and metadata, in a structured format for later analysis.
    """
    
    def __init__(self):
        """Initialize the DataLoggerChain."""
        pass
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Log execution results to the agent's state.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Execution MAGI: 5. Logging Data ---")
        
        try:
            # Get execution data from state
            log = state.get("execution_log", "No log available.")
            output = state.get("simulation_output", {})
            error = state.get("error")
            
            # Create an execution record
            execution_record = ExecutionRecord(
                log=log,
                output_data=output,
                success=error is None,
                error=error,
                metadata={
                    "source": "execution_magi",
                    "task_description": state.get("current_task_description", "N/A"),
                    "code_approved": state.get("code_approved", False)
                }
            )
            
            # Get existing execution results or initialize a new list
            execution_results = state.get("execution_results", [])
            if not isinstance(execution_results, list):
                execution_results = []
            
            # Add the new record
            execution_results.append(execution_record.model_dump())
            
            print("  > Execution results have been logged successfully.")
            
            return Command(
                update={
                    "execution_results": execution_results,
                    "last_execution": execution_record.model_dump(),
                    "messages": [{"role": "system", "content": "Execution results have been logged."}]
                }
            )
            
        except Exception as e:
            error_msg = f"Error logging execution results: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "error": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, state: AgentState) -> ExecutionRecord:
        """
        Create an execution record from the current state.
        
        Args:
            state: The current agent state
            
        Returns:
            ExecutionRecord: The created execution record
        """
        log = state.get("execution_log", "No log available.")
        output = state.get("simulation_output", {})
        error = state.get("error")
        
        return ExecutionRecord(
            log=log,
            output_data=output,
            success=error is None,
            error=error,
            metadata={
                "source": "execution_magi",
                "task_description": state.get("current_task_description", "N/A"),
                "code_approved": state.get("code_approved", False),
                "timestamp": state.get("timestamp")
            }
        )


# Create a singleton instance
data_logger_chain_instance = DataLoggerChain()


async def data_logger_chain(state: AgentState) -> Command:
    """
    Entry point for the data logger chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await data_logger_chain_instance(state)

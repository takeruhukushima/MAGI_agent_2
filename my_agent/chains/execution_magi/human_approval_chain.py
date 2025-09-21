from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.types import Command

from my_agent.utils.state import AgentState

class ApprovalStatus(str, Enum):
    """Possible statuses for human approval."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ApprovalRequest(BaseModel):
    """A request for human approval."""
    status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING,
        description="The current status of the approval request."
    )
    code: str = Field(
        ...,
        description="The code that requires approval."
    )
    reason: Optional[str] = Field(
        None,
        description="Optional reason for approval or rejection."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the approval request."
    )


class HumanApprovalChain:
    """
    Implements the 'Manus' concept - a node that waits for human approval.
    
    This chain pauses the execution flow and waits for human intervention
    to approve or reject the generated code before proceeding with execution.
    """
    
    def __init__(self, require_approval: bool = True):
        """
        Initialize the HumanApprovalChain.
        
        Args:
            require_approval: If False, will automatically approve all requests.
                             Useful for testing or non-interactive environments.
        """
        self.require_approval = require_approval
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Request human approval for the generated code.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Execution MAGI: 3. Awaiting Human Approval ---")
        
        # Get the generated code from state
        generated_code = state.get("generated_code", "# No code was generated.")
        
        # Create an approval request
        approval_request = ApprovalRequest(
            code=generated_code,
            metadata={
                "source": "code_generator",
                "timestamp": state.get("timestamp"),
                "task_description": state.get("current_task_description", "N/A")
            }
        )
        
        if not self.require_approval:
            print("  > Auto-approval is enabled. Proceeding with execution.")
            approval_request.status = ApprovalStatus.APPROVED
            approval_request.reason = "Auto-approved in non-interactive mode."
            return Command(
                update={
                    "approval_request": approval_request.model_dump(),
                    "code_approved": True,
                    "messages": [{"role": "system", "content": "Code auto-approved in non-interactive mode."}]
                }
            )
        
        # In a real implementation, this would trigger a UI prompt for human approval
        print("\n" + "="*60)
        print("ACTION REQUIRED: HUMAN INTERVENTION (MANUS)")
        print("="*60)
        print("The following code has been generated for execution.")
        print("Please review the code and provide your approval to continue.")
        print("-" * 60)
        print(generated_code[:1000] + ("..." if len(generated_code) > 1000 else ""))
        print("-" * 60)
        
        # In a real implementation, we would wait for actual user input here
        # For now, we'll simulate a pending approval that needs to be handled by the agent
        return Command(
            update={
                "approval_request": approval_request.model_dump(),
                "code_approved": False,
                "messages": [{"role": "system", "content": "Waiting for human approval of the generated code."}]
            },
            # This would trigger a pause in the graph execution
            pause=True,
            # Metadata for the UI to know how to handle this pause
            metadata={
                "type": "human_approval",
                "message": "Please review and approve the generated code.",
                "code_preview": generated_code[:500],
                "requires_action": True
            }
        )
    
    async def approve(self, state: AgentState, reason: str = None) -> Command:
        """Approve the current code execution."""
        approval_request = state.get("approval_request", {})
        if isinstance(approval_request, dict):
            approval_request["status"] = ApprovalStatus.APPROVED
            if reason:
                approval_request["reason"] = reason
        
        return Command(
            update={
                "approval_request": approval_request,
                "code_approved": True,
                "messages": [{"role": "user", "content": f"Code approved. {reason or ''}".strip()}]
            }
        )
    
    async def reject(self, state: AgentState, reason: str) -> Command:
        """Reject the current code execution."""
        approval_request = state.get("approval_request", {})
        if isinstance(approval_request, dict):
            approval_request["status"] = ApprovalStatus.REJECTED
            approval_request["reason"] = reason
        
        return Command(
            update={
                "approval_request": approval_request,
                "code_approved": False,
                "error": f"Code execution rejected: {reason}",
                "messages": [{"role": "user", "content": f"Code rejected: {reason}"}]
            }
        )


# Create a singleton instance
human_approval_chain_instance = HumanApprovalChain()


async def human_approval_chain(state: AgentState) -> Command:
    """
    Entry point for the human approval chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await human_approval_chain_instance(state)

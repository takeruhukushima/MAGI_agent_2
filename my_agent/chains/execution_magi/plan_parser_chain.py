from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from my_agent.utils.state import AgentState
from my_agent.prompts import ExecutionPrompts
from my_agent.settings import settings

class ExecutableTask(BaseModel):
    """A single, executable task parsed from the research plan."""
    task_id: int = Field(..., description="A unique identifier for the task.")
    description: str = Field(..., description="A clear, concise description of the task to be performed.")
    required_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of parameters needed for the simulation or experiment."
    )

    @field_validator('description')
    @classmethod
    def validate_description_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Task description cannot be empty")
        return v.strip()


class ParsedPlan(BaseModel):
    """A list of executable tasks parsed from the research plan."""
    tasks: List[ExecutableTask] = Field(
        default_factory=list,
        description="The list of tasks to be executed."
    )


class PlanParserChain:
    """
    TeX形式の研究計画書を解析し、実行可能なタスクリストに変換するスキルクラス。
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the PlanParserChain with an optional LLM."""
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature
        )
        
        # Use the prompt template directly from ExecutionPrompts
        self.prompt_template = ChatPromptTemplate.from_template(
            template=ExecutionPrompts.PLAN_PARSER,
            template_format="f-string"
        )
        
        # Create the chain with structured output
        self.chain = self.prompt_template | self.llm.with_structured_output(ParsedPlan)
    
    def _format_messages(self, research_plan_tex: str) -> List[Dict[str, str]]:
        """Format messages for the LLM."""
        return [{"role": "user", "content": self.prompt_template.format(
            research_plan_tex=research_plan_tex
        )}]
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Parse the research plan into executable tasks.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Execution MAGI: 1. Parsing Research Plan ---")
        
        research_plan_tex = state.get("research_plan_tex")
        if not research_plan_tex:
            print("  > No research plan found in state")
            return Command(
                update={
                    "parsed_plan": [],
                    "messages": [{"role": "system", "content": "Error: No research plan available to parse."}]
                }
            )
        
        try:
            # Parse the plan
            parsed_plan = await self.run(research_plan_tex, state.get("messages", []))
            
            return Command(
                update={
                    "parsed_plan": [task.model_dump() for task in parsed_plan.tasks],
                    "messages": [{"role": "system", "content": f"Successfully parsed research plan into {len(parsed_plan.tasks)} executable tasks."}]
                }
            )
            
        except Exception as e:
            error_msg = f"Error parsing research plan: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "parsed_plan": [],
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, research_plan_tex: str, messages: Optional[List[Dict[str, str]]] = None) -> ParsedPlan:
        """
        Parse the research plan into executable tasks.
        
        Args:
            research_plan_tex: The research plan in TeX format
            messages: Previous conversation messages for context
            
        Returns:
            ParsedPlan: The parsed plan with executable tasks
        """
        # Format the messages
        formatted_messages = self._format_messages(research_plan_tex)
        
        # Add previous messages for context if available
        if messages:
            formatted_messages = messages + formatted_messages
        
        # Parse the plan
        parsed_plan = await self.chain.ainvoke({"research_plan_tex": research_plan_tex})
        
        print(f"  > Plan parsed into {len(parsed_plan.tasks)} tasks.")
        return parsed_plan


# Create a singleton instance
plan_parser_chain_instance = PlanParserChain()


async def plan_parser_chain(state: AgentState) -> Command:
    """
    Entry point for the plan parser chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await plan_parser_chain_instance(state)

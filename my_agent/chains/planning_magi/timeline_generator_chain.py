from pathlib import Path
from typing import Dict, Any, List, Literal, Optional
from datetime import date, timedelta

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field, validator

from my_agent.prompts import PromptManager




class Milestone(BaseModel):
    """A significant event or deliverable in the project timeline."""
    name: str = Field(..., description="Name of the milestone")
    description: str = Field(..., description="Detailed description of the milestone")
    due_date: str = Field(..., description="Due date in YYYY-MM-DD format")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of milestone names this milestone depends on"
    )
    priority: str = Field(
        "medium",
        description="Priority level: low, medium, high, or critical"
    )
    status: str = Field(
        "not_started",
        description="Current status: not_started, in_progress, completed, or blocked"
    )

    @validator('due_date')
    def validate_date_format(cls, v):
        """Validate the date format."""
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class ProjectTimeline(BaseModel):
    """Complete project timeline with milestones and dependencies."""
    project_name: str = Field(..., description="Name of the project")
    start_date: str = Field(..., description="Project start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="Project end date in YYYY-MM-DD format")
    milestones: List[Milestone] = Field(
        ...,
        description="List of project milestones",
        min_items=3
    )
    buffer_days: int = Field(
        7,
        description="Number of buffer days to include in the timeline",
        ge=0
    )
    critical_path: List[str] = Field(
        ...,
        description="List of milestone names that form the critical path"
    )

    @validator('end_date')
    def validate_end_date(cls, v, values):
        """Validate that end date is after start date."""
        if 'start_date' in values and v:
            start_date = date.fromisoformat(values['start_date'])
            end_date = date.fromisoformat(v)
            if end_date < start_date:
                raise ValueError("End date must be after start date")
        return v


class TimelineGeneratorChain:
    """
    A chain that generates a detailed project timeline with milestones based on the experimental design.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('planning', 'TIMELINE_GENERATOR')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            ProjectTimeline,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Command[Literal["tex_formatter"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Planning MAGI: 4. Generating Project Timeline ---")
        
        # Get the necessary data from state
        experimental_design = state.get("experimental_design", {})
        research_goal = state.get("research_goal", {})
        
        try:
            # Generate the project timeline
            timeline = self.run(
                experimental_design=experimental_design,
                research_goal=research_goal,
                messages=state.get("messages", [])
            )
            
            print(f"  > Timeline generated with {len(timeline.milestones)} milestones")
            
            return Command(
                goto="tex_formatter",
                update={
                    "project_timeline": timeline.dict(),
                    "status": "timeline_generated"
                }
            )
            
        except Exception as e:
            print(f"  > Error generating timeline: {e}")
            return Command(
                goto="tex_formatter",
                update={
                    "status": "timeline_generation_failed",
                    "error": str(e)
                }
            )
    
    def run(
        self,
        experimental_design: Dict[str, Any],
        research_goal: Dict[str, Any],
        messages: Optional[list[BaseMessage]] = None
    ) -> ProjectTimeline:
        """Generate a project timeline based on the experimental design."""
        try:
            # Format messages for context if available
            conversation_context = self._format_messages(messages) if messages else ""
            
            # Generate the project timeline
            return self.chain.invoke({
                "experimental_design": str(experimental_design),
                "research_goal": str(research_goal),
                "current_date": date.today().isoformat(),
                "conversation_context": conversation_context
            })
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate timeline: {str(e)}")
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for the prompt context."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# Create a single instance of the chain
from my_agent.settings import settings

timeline_generator_chain = TimelineGeneratorChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)
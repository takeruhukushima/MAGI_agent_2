from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field, validator
from datetime import date, timedelta

from my_agent.prompts import PromptManager




class ExperimentalGroup(BaseModel):
    """A group in an experimental design."""
    name: str = Field(..., description="Name of the group")
    description: str = Field(..., description="Description of the group")
    sample_size: int = Field(..., description="Number of samples in this group", gt=0)
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables and their values for this group"
    )


class ExperimentalProcedure(BaseModel):
    """A step in the experimental procedure."""
    step_number: int = Field(..., description="Step number in the procedure", ge=1)
    description: str = Field(..., description="Detailed description of this step")
    duration: str = Field(..., description="Estimated duration of this step")
    materials: List[str] = Field(
        default_factory=list,
        description="List of materials or equipment needed for this step"
    )


class ExperimentalDesign(BaseModel):
    """Complete experimental design for a research study."""
    title: str = Field(..., description="Title of the experimental design")
    objective: str = Field(..., description="Objective of the experiment")
    hypothesis: str = Field(..., description="Testable hypothesis")
    variables: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary of independent and dependent variables"
    )
    groups: List[ExperimentalGroup] = Field(
        ...,
        description="Experimental and control groups",
        min_items=1
    )
    procedure: List[ExperimentalProcedure] = Field(
        ...,
        description="Step-by-step experimental procedure",
        min_items=3
    )
    data_collection_methods: List[str] = Field(
        ...,
        description="Methods for collecting data",
        min_items=1
    )
    analysis_methods: List[str] = Field(
        ...,
        description="Methods for analyzing the collected data",
        min_items=1
    )
    timeline: Dict[str, str] = Field(
        ...,
        description="Estimated timeline for the experiment"
    )
    ethical_considerations: List[str] = Field(
        default_factory=list,
        description="Ethical considerations and approvals needed"
    )

    @validator('timeline')
    def validate_timeline(cls, v):
        """Validate that the timeline has start and end dates."""
        if 'start_date' not in v or 'end_date' not in v:
            raise ValueError("Timeline must include 'start_date' and 'end_date'")
        return v


class ExperimentalDesignChain:
    """
    A chain that designs detailed experimental plans based on research goals and methodology.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('planning', 'EXPERIMENTAL_DESIGN')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            ExperimentalDesign,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Command[Literal["timeline_generator"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Planning MAGI: 3. Designing Experiment ---")
        
        # Get the necessary data from state
        research_goal = state.get("research_goal", {})
        methodology = state.get("selected_methodology", {})
        
        try:
            # Generate the experimental design
            design = self.run(
                research_goal=research_goal,
                methodology=methodology,
                messages=state.get("messages", [])
            )
            
            print(f"  > Experimental design created: {design.title}")
            
            return Command(
                goto="timeline_generator",
                update={
                    "experimental_design": design.dict(),
                    "status": "experiment_designed"
                }
            )
            
        except Exception as e:
            print(f"  > Error designing experiment: {e}")
            return Command(
                goto="timeline_generator",
                update={
                    "status": "experiment_design_failed",
                    "error": str(e)
                }
            )
    
    def run(
        self,
        research_goal: Dict[str, Any],
        methodology: Dict[str, Any],
        messages: Optional[list[BaseMessage]] = None
    ) -> ExperimentalDesign:
        """Generate an experimental design based on the research goal and methodology."""
        try:
            # Format messages for context if available
            conversation_context = self._format_messages(messages) if messages else ""
            
            # Generate the experimental design
            return self.chain.invoke({
                "research_goal": str(research_goal),
                "methodology": str(methodology),
                "current_date": date.today().isoformat(),
                "conversation_context": conversation_context
            })
            
        except Exception as e:
            raise RuntimeError(f"Failed to design experiment: {str(e)}")
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for the prompt context."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# Create a single instance of the chain
from my_agent.settings import settings

experimental_design_chain = ExperimentalDesignChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

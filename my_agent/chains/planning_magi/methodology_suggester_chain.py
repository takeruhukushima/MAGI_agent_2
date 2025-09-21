from typing import Dict, Any, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.prompts import PromptManager


class MethodologyOption(BaseModel):
    """A single methodology option with its details."""
    name: str = Field(..., description="Name of the methodology")
    description: str = Field(..., description="Detailed description of the methodology")
    advantages: List[str] = Field(
        ...,
        description="List of advantages of this methodology",
        min_items=2,
        max_items=5
    )
    limitations: List[str] = Field(
        ...,
        description="List of limitations or challenges of this methodology",
        min_items=1,
        max_items=3
    )
    suitability: int = Field(
        ...,
        description="Suitability score for the research goal (1-10)",
        ge=1,
        le=10
    )


class MethodologySuggestion(BaseModel):
    """A collection of methodology suggestions for a research goal."""
    research_goal: str = Field(..., description="The research goal these methodologies address")
    methodologies: List[MethodologyOption] = Field(
        ...,
        description="List of suggested methodologies",
        min_items=2,
        max_items=4
    )
    recommendation: str = Field(
        ...,
        description="Detailed recommendation on which methodology to choose and why"
    )


class MethodologySuggesterChain:
    """
    A chain that suggests appropriate research methodologies based on the research goal.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('planning', 'METHODOLOGY_SUGGESTER')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            MethodologySuggestion,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Command[Literal["experimental_design"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Planning MAGI: 2. Suggesting Methodologies ---")
        
        # Get the research goal from state
        research_goal = state.get("research_goal", {})
        
        try:
            # Generate methodology suggestions
            suggestions = self.run(
                research_goal=research_goal,
                messages=state.get("messages", [])
            )
            
            print(f"  > Suggested {len(suggestions.methodologies)} methodologies")
            
            # Get the recommended methodology (highest suitability score)
            recommended = max(
                suggestions.methodologies,
                key=lambda m: m.suitability,
                default=None
            )
            
            return Command(
                goto="experimental_design",
                update={
                    "methodology_suggestions": suggestions.dict(),
                    "selected_methodology": recommended.dict() if recommended else None,
                    "status": "methodologies_suggested"
                }
            )
            
        except Exception as e:
            print(f"  > Error suggesting methodologies: {e}")
            return Command(
                goto="experimental_design",
                update={
                    "status": "methodology_suggestion_failed",
                    "error": str(e)
                }
            )
    
    def run(
        self,
        research_goal: Dict[str, Any],
        messages: Optional[list[BaseMessage]] = None
    ) -> MethodologySuggestion:
        """Generate methodology suggestions based on the research goal."""
        try:
            # Format messages for context if available
            conversation_context = self._format_messages(messages) if messages else ""
            
            # Generate methodology suggestions
            return self.chain.invoke({
                "research_goal": str(research_goal),
                "conversation_context": conversation_context
            })
            
        except Exception as e:
            raise RuntimeError(f"Failed to suggest methodologies: {str(e)}")
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for the prompt context."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# Create a single instance of the chain
from my_agent.settings import settings

methodology_suggester_chain = MethodologySuggesterChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

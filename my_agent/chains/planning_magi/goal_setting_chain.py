from typing import Dict, Any, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.prompts import PromptManager


class ResearchGoal(BaseModel):
    """A well-defined research goal with key components."""
    title: str = Field(..., description="A concise title for the research goal")
    description: str = Field(..., description="Detailed description of the research goal")
    objectives: list[str] = Field(
        ...,
        description="List of specific, measurable objectives to achieve this goal",
        min_items=3,
        max_items=5
    )
    expected_outcomes: list[str] = Field(
        ...,
        description="List of expected outcomes or deliverables",
        min_items=2,
        max_items=4
    )
    success_metrics: list[str] = Field(
        ...,
        description="List of metrics to measure the success of this research",
        min_items=2,
        max_items=4
    )


class GoalSettingChain:
    """
    A chain that sets clear, actionable research goals based on survey findings.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('planning', 'GOAL_SETTING')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            ResearchGoal,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Command[Literal["methodology_suggester"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Planning MAGI: 1. Setting Research Goal ---")
        
        # Get the necessary data from state
        survey_summary = state.get("survey_summary", {})
        research_theme = state.get("research_theme", "")
        
        try:
            # Generate the research goal
            goal = self.run(
                research_theme=research_theme,
                survey_summary=survey_summary,
                messages=state.get("messages", [])
            )
            
            print(f"  > Research goal set: {goal.title}")
            
            return Command(
                goto="methodology_suggester",
                update={
                    "research_goal": goal.dict(),
                    "status": "goal_set"
                }
            )
            
        except Exception as e:
            print(f"  > Error setting research goal: {e}")
            # Create a default goal structure if generation fails
            default_goal = ResearchGoal(
                title=f"Research on {research_theme}",
                description=f"Conduct a comprehensive study on {research_theme} based on survey findings.",
                objectives=[
                    f"Investigate key aspects of {research_theme}",
                    "Analyze existing research and identify gaps",
                    "Develop new insights or solutions"
                ],
                expected_outcomes=[
                    "Detailed research report",
                    "Analysis of findings"
                ],
                success_metrics=[
                    "Completeness of literature review",
                    "Novelty of insights"
                ]
            )
            
            return Command(
                goto="methodology_suggester",
                update={
                    "research_goal": default_goal.dict(),
                    "status": "default_goal_used"
                }
            )
    
    def run(
        self,
        research_theme: str,
        survey_summary: Dict[str, Any],
        messages: Optional[list[BaseMessage]] = None
    ) -> ResearchGoal:
        """Generate a research goal based on the survey findings."""
        try:
            # Format messages for context if available
            conversation_context = self._format_messages(messages) if messages else ""
            
            # Generate the research goal
            return self.chain.invoke({
                "research_theme": research_theme,
                "survey_summary": str(survey_summary),
                "conversation_context": conversation_context
            })
            
        except Exception as e:
            raise RuntimeError(f"Failed to set research goal: {str(e)}")
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for the prompt context."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# Create a single instance of the chain
from my_agent.settings import settings

goal_setting_chain = GoalSettingChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

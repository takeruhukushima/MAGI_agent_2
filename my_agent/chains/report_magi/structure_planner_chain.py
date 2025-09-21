from typing import List, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.settings import settings
from my_agent.prompts import PromptManager


class PaperStructure(BaseModel):
    """The structure of the academic paper."""
    title: str = Field(..., description="The title of the paper")
    sections: List[dict] = Field(
        ...,
        description="List of sections with their titles and descriptions"
    )


class StructurePlannerChain:
    """
    A chain that plans the structure of an academic paper.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt = PromptManager.get_prompt("report", "STRUCTURE_PLANNER")
        self.prompt = ChatPromptTemplate.from_template(prompt)
        self.chain = self.prompt | self.llm.with_structured_output(
            PaperStructure,
            method="function_calling"
        )
    
    def __call__(self, state: dict) -> Command[Literal["section_writer"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Report MAGI: 1. Planning Paper Structure ---")
        
        research_goal = state.get("research_goal") or "N/A"
        
        try:
            structure = self.run(research_goal, state.get("messages", []))
            print(f"  > Paper structure planned with {len(structure.sections)} sections")
            
            return Command(
                goto="section_writer",
                update={
                    "paper_structure": structure.dict(),
                    "current_section_index": 0  # Start with the first section
                }
            )
            
        except Exception as e:
            print(f"  > Error during structure planning: {e}")
            # Fallback to a basic structure
            fallback_structure = PaperStructure(
                title=research_goal,
                sections=[
                    {"title": "Introduction", "description": "Introduce the research topic and objectives"},
                    {"title": "Methods", "description": "Describe the research methodology"},
                    {"title": "Results", "description": "Present the research findings"},
                    {"title": "Discussion", "description": "Discuss the implications of the results"},
                    {"title": "Conclusion", "description": "Summarize the key findings and future work"}
                ]
            )
            return Command(
                goto="section_writer",
                update={
                    "paper_structure": fallback_structure.dict(),
                    "current_section_index": 0
                }
            )
    
    def run(self, research_goal: str, messages: list[BaseMessage] = None) -> PaperStructure:
        """Generate a paper structure based on the research goal."""
        try:
            conversation_history = self._format_messages(messages) if messages else ""
            return self.chain.invoke({
                "research_goal": research_goal,
                "conversation_history": conversation_history
            })
        except Exception as e:
            raise RuntimeError(f"Structure planning failed: {str(e)}")
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for the prompt."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# Create a single instance of the chain
structure_planner_chain = StructurePlannerChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

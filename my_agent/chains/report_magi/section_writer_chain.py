from typing import List, Literal, Dict, Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.settings import settings
from my_agent.prompts import PromptManager


class PaperSection(BaseModel):
    """A single section of an academic paper."""
    title: str = Field(..., description="The title of the section")
    content: str = Field(..., description="The content of the section in markdown format")
    references: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of references used in this section with 'title' and 'url'"
    )


class SectionWriterChain:
    """
    A chain that writes individual sections of an academic paper.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt = PromptManager.get_prompt("report", "SECTION_WRITER")
        self.prompt = ChatPromptTemplate.from_template(prompt)
    
    def __call__(self, state: dict) -> Command[Literal["section_writer", "content_aggregator"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Report MAGI: 2. Writing Paper Section ---")
        
        paper_structure = state.get("paper_structure", {})
        sections = paper_structure.get("sections", [])
        current_idx = state.get("current_section_index", 0)
        
        if current_idx >= len(sections):
            print("  > All sections have been written.")
            return Command(
                goto="content_aggregator",
                update={"status": "all_sections_written"}
            )
        
        current_section = sections[current_idx]
        print(f"  > Writing section: {current_section.get('title', f'Section {current_idx + 1}')}")
        
        try:
            section_content = self.run(
                section_title=current_section.get("title", ""),
                section_description=current_section.get("description", ""),
                paper_title=paper_structure.get("title", "Research Paper"),
                previous_sections=sections[:current_idx],
                research_data=state.get("research_data", {})
            )
            
            # Update the section with the generated content
            sections[current_idx].update(section_content.dict())
            
            # Determine if there are more sections to write
            next_node = "section_writer" if current_idx + 1 < len(sections) else "content_aggregator"
            
            return Command(
                goto=next_node,
                update={
                    "paper_structure": {
                        **paper_structure,
                        "sections": sections
                    },
                    "current_section_index": current_idx + 1,
                    "status": "section_written"
                }
            )
            
        except Exception as e:
            print(f"  > Error writing section: {e}")
            # Skip to the next section on error
            return Command(
                goto="section_writer" if current_idx + 1 < len(sections) else "content_aggregator",
                update={
                    "current_section_index": current_idx + 1,
                    "status": f"error_writing_section_{current_idx}"
                }
            )
    
    def run(
        self,
        section_title: str,
        section_description: str,
        paper_title: str,
        previous_sections: List[dict],
        research_data: dict
    ) -> PaperSection:
        """Generate content for a specific section."""
        try:
            chain = self.prompt | self.llm.with_structured_output(
                PaperSection,
                method="function_calling"
            )
            
            return chain.invoke({
                "section_title": section_title,
                "section_description": section_description,
                "paper_title": paper_title,
                "previous_sections": previous_sections,
                "research_data": research_data
            })
            
        except Exception as e:
            raise RuntimeError(f"Failed to write section '{section_title}': {str(e)}")


# Create a single instance of the chain
section_writer_chain = SectionWriterChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

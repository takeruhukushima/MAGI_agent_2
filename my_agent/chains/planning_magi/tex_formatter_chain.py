from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from my_agent.utils.state import AgentState
from my_agent.prompts import PromptManager
from my_agent.settings import settings

class ResearchPlanTex(BaseModel):
    """A structured representation of a research plan in TeX format."""
    tex_content: str = Field(
        ...,
        description="The complete TeX document content for the research plan."
    )
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary of section names to their TeX content."
    )


class TexFormatterChain:
    """
    A chain that formats all planning elements into a well-structured TeX document.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the TexFormatterChain with an optional LLM."""
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature
        )
        
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('report', 'TEX_COMPILER')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the chain with structured output
        self.chain = self.prompt | self.llm
    
    def _get_planning_elements(self, state: AgentState) -> Dict[str, str]:
        """Extract planning elements from the state."""
        return {
            "research_goal": state.get("research_goal", "N/A"),
            "methodology": state.get("methodology", "N/A"),
            "experimental_design": state.get("experimental_design", "N/A"),
            "timeline": state.get("timeline", "N/A")
        }
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Format all planning elements into a TeX document.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Planning MAGI: 5. Formatting TeX Report ---")
        
        try:
            # Format the TeX document
            result = await self.run(state)
            
            if result.tex_content:
                print("  > TeX content generated successfully.")
                return Command(
                    update={
                        "research_plan_tex": result.tex_content,
                        "messages": [{"role": "system", "content": "Successfully generated TeX report."}]
                    }
                )
            else:
                error_msg = "Failed to generate TeX content."
                print(f"  > {error_msg}")
                return Command(
                    update={
                        "research_plan_tex": "% Failed to generate TeX report.",
                        "error": error_msg,
                        "messages": [{"role": "system", "content": error_msg}]
                    }
                )
                
        except Exception as e:
            error_msg = f"Error formatting TeX: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "research_plan_tex": f"% Error generating TeX: {str(e)}",
                    "error": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, state: AgentState) -> ResearchPlanTex:
        """
        Format all planning elements into a TeX document.
        
        Args:
            state: The current agent state
            
        Returns:
            ResearchPlanTex: The formatted TeX document
        """
        try:
            # Format the input for the prompt
            formatted_input = self._get_planning_elements(state)
            
            # Generate the TeX content
            response = self.chain.invoke(formatted_input)
            tex_content = response.content
            
            # Extract sections if possible (this is a simple example)
            sections = {}
            if "\section" in tex_content:
                # This is a simplified section extraction
                import re
                section_matches = re.finditer(r'\\(section|subsection|subsubsection)\*?\{(.*?)\}(.*?)(?=\\section|\Z)', 
                                            tex_content, 
                                            re.DOTALL)
                
                for match in section_matches:
                    section_type = match.group(1)
                    section_title = match.group(2).strip()
                    section_content = match.group(3).strip()
                    sections[f"{section_type}:{section_title}"] = section_content
            
            return ResearchPlanTex(
                tex_content=tex_content,
                sections=sections
            )
            
        except Exception as e:
            # Return a minimal error document if generation fails
            error_content = (
                "% Error generating TeX document\n"
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                f"% Error: {str(e)}\n"
                "Failed to generate TeX document.\n"
                "\\end{document}"
            )
            return ResearchPlanTex(
                tex_content=error_content,
                sections={"error": str(e)}
            )


# Create a single instance of the chain
from my_agent.settings import settings

tex_formatter_chain = TexFormatterChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)
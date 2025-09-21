from typing import Dict, Any, Literal, Optional
import re

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.settings import settings
from my_agent.prompts import PromptManager


class LatexDocument(BaseModel):
    """A LaTeX document with its content and metadata."""
    content: str = Field(..., description="The LaTeX document content")
    title: str = Field(..., description="The title of the document")
    author: str = Field("MAGI System", description="The author of the document")
    date: str = Field("\\today", description="The document date")
    document_class: str = Field("article", description="The LaTeX document class")
    packages: Dict[str, str] = Field(
        default_factory=lambda: {
            "babel": "english",
            "graphicx": "",
            "hyperref": "",
            "amsmath": "",
            "amssymb": ""
        },
        description="LaTeX packages to include with their options"
    )


class TexGeneratorChain:
    """
    A chain that generates a complete LaTeX document from structured content.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt = PromptManager.get_prompt("report", "TEX_COMPILER")
        self.prompt = ChatPromptTemplate.from_template(prompt)
        self.chain = self.prompt | self.llm
    
    def __call__(self, state: dict) -> Command[Literal["final_review"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Report MAGI: 5. Generating LaTeX Document ---")
        
        paper_structure = state.get("paper_structure", {})
        aggregated_content = state.get("aggregated_content", {})
        
        try:
            latex_doc = self.run(
                title=paper_structure.get("title", "Research Paper"),
                sections=paper_structure.get("sections", []),
                content=aggregated_content
            )
            
            print("  > LaTeX document generated successfully.")
            
            return Command(
                goto="final_review",
                update={
                    "latex_document": latex_doc.dict(),
                    "status": "latex_generated"
                }
            )
            
        except Exception as e:
            print(f"  > Error generating LaTeX: {e}")
            return Command(
                goto="final_review",
                update={
                    "status": "latex_generation_failed",
                    "error": str(e)
                }
            )
    
    def run(
        self,
        title: str,
        sections: list[dict],
        content: dict
    ) -> LatexDocument:
        """Generate a LaTeX document from the given content."""
        try:
            # First, get the raw LaTeX content from the LLM
            response = self.chain.invoke({
                "title": title,
                "sections": sections,
                "content": content
            })
            
            # Extract LaTeX content (handling both raw LaTeX and markdown code blocks)
            latex_content = response.content.strip()
            
            # Extract LaTeX from code blocks if present
            if "```latex" in latex_content:
                latex_content = re.search(r"```(?:latex)?\n(.*?)\n```", latex_content, re.DOTALL)
                if latex_content:
                    latex_content = latex_content.group(1)
            
            # Create a basic document structure if the response is just the content
            if not latex_content.startswith("\\documentclass"):
                latex_content = fr"""\documentclass{{article}}
\usepackage[english]{{babel}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}

\title{{{title}}}
\author{{MAGI System}}
\date{{\today}}

\begin{{document}}
\maketitle

{latex_content}

\end{{document}}"""
            
            return LatexDocument(
                content=latex_content,
                title=title,
                packages={
                    "babel": "english",
                    "graphicx": "",
                    "hyperref": "",
                    "amsmath": "",
                    "amssymb": ""
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate LaTeX document: {str(e)}")


# Create a single instance of the chain
tex_generator_chain = TexGeneratorChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

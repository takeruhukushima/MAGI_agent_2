from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from my_agent.utils.state import AgentState
from my_agent.prompts import ExecutionPrompts
from my_agent.prompts import load_prompt
from my_agent.settings import settings

class CodeGenerationResult(BaseModel):
    """Result of code generation."""
    success: bool = Field(..., description="Whether the code generation was successful.")
    generated_code: str = Field(
        default="",
        description="The generated Python code for the simulation or analysis."
    )
    error: Optional[str] = Field(
        None,
        description="Error message if code generation failed."
    )
    task_description: Optional[str] = Field(
        None,
        description="The task description that the code was generated for."
    )


class CodeGeneratorChain:
    """
    Chain for generating Python code for simulations and analyses based on parsed tasks.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the CodeGeneratorChain with an optional LLM."""
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature
        )
        
        # Create the prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            template=ExecutionPrompts.CODE_GENERATOR,
            template_format="f-string"
        )
        
        # Create the chain
        self.chain = self.prompt_template | self.llm
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        import re
        
        # Try to find Python code blocks
        python_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        # Fallback to any code block
        any_blocks = re.findall(r'```\n(.*?)```', text, re.DOTALL)
        if any_blocks:
            return any_blocks[0].strip()
        
        # If no code blocks, return the text as is
        return text.strip()
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Generate code for the first task in the parsed plan.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Execution MAGI: 2. Generating Code ---")
        
        parsed_plan = state.get("parsed_plan", [])
        if not parsed_plan:
            error_msg = "No tasks found in the plan to generate code for."
            print(f"  > {error_msg}")
            return Command(
                update={
                    "generated_code": "# No tasks found in the plan.",
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
        
        # Get the first task
        first_task = parsed_plan[0]
        
        # Safely extract task description
        if hasattr(first_task, 'description'):
            task_description = first_task.description
        elif isinstance(first_task, dict):
            task_description = first_task.get('description', 'No description provided.')
        else:
            task_description = 'No description provided.'
        
        try:
            # Generate code for the task
            result = await self.run(task_description, state.get("messages", []))
            
            if result.success:
                print("  > Code generated successfully.")
                return Command(
                    update={
                        "generated_code": result.generated_code,
                        "current_task_description": task_description,
                        "messages": [{"role": "system", "content": "Code generated successfully."}]
                    }
                )
            else:
                error_msg = f"Failed to generate code: {result.error}"
                print(f"  > {error_msg}")
                return Command(
                    update={
                        "generated_code": f"# Failed to generate code: {result.error}",
                        "error": error_msg,
                        "messages": [{"role": "system", "content": error_msg}]
                    }
                )
                
        except Exception as e:
            error_msg = f"Error during code generation: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "generated_code": f"# Error during code generation: {str(e)}",
                    "error": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, task_description: str, messages: Optional[List[Dict[str, str]]] = None) -> CodeGenerationResult:
        """
        Generate code for the given task description.
        
        Args:
            task_description: Description of the task to generate code for
            messages: Previous conversation messages for context
            
        Returns:
            CodeGenerationResult: The result of the code generation
        """
        try:
            # Generate the prompt
            prompt = self.prompt_template.format(task_description=task_description)
            
            # Get the LLM response
            response = await self.chain.ainvoke({"task_description": task_description})
            
            # Extract code from the response
            generated_code = self._extract_code_from_markdown(response.content)
            
            if not generated_code:
                return CodeGenerationResult(
                    success=False,
                    generated_code="# Failed to extract code from the response.",
                    error="No code block found in the response.",
                    task_description=task_description
                )
            
            print("  > Code generation completed.")
            return CodeGenerationResult(
                success=True,
                generated_code=generated_code,
                task_description=task_description
            )
            
        except Exception as e:
            return CodeGenerationResult(
                success=False,
                generated_code=f"# Failed to generate code: {str(e)}",
                error=str(e),
                task_description=task_description
            )


# Create a singleton instance
code_generator_chain_instance = CodeGeneratorChain()


async def code_generator_chain(state: AgentState) -> Command:
    """
    Entry point for the code generator chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await code_generator_chain_instance(state)

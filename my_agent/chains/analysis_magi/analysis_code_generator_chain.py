from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, field_validator
import re
import json
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from my_agent.utils.state import AgentState
from my_agent.prompts import AnalysisPrompts
from my_agent.settings import settings

class AnalysisCode(BaseModel):
    """Structured representation of generated analysis code."""
    code: str = Field(
        ...,
        description="The Python code for analysis and visualization."
    )
    description: str = Field(
        ...,
        description="A brief description of what the code does."
    )
    dependencies: List[str] = Field(
        default_factory=lambda: ["pandas", "numpy", "matplotlib", "seaborn"],
        description="List of Python package dependencies required by the code."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters that can be adjusted in the analysis."
    )
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="List of expected output artifacts (e.g., figures, tables)."
    )


class AnalysisCodeGeneratorChain:
    """
    A chain that generates Python code for data analysis and visualization
    based on the selected analysis method.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the AnalysisCodeGeneratorChain with an optional LLM."""
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.model.google_gemini_fast_model,
            temperature=settings.model.temperature
        )
        
        # Use the prompt template directly from AnalysisPrompts
        self.prompt_template = ChatPromptTemplate.from_template(
            template=AnalysisPrompts.ANALYSIS_CODE_GENERATOR,
            template_format="f-string"
        )
        
        # Create the chain with structured output
        self.chain = self.prompt_template | self.llm.with_structured_output(AnalysisCode)
    
    def _prepare_context(self, state: AgentState) -> Dict[str, Any]:
        """Prepare the context for code generation from the agent state."""
        # Get analysis method details
        analysis_method = state.get("primary_method", {})
        if isinstance(analysis_method, dict):
            analysis_method = analysis_method.get("name", "Perform a basic analysis.")
        
        # Get data from execution results
        execution_results = state.get("execution_results", {})
        if isinstance(execution_results, list) and len(execution_results) > 0:
            # Get the most recent execution result
            execution_results = execution_results[-1]
        
        # Extract dataset
        dataset = {}
        if isinstance(execution_results, dict):
            if 'output_data' in execution_results and 'dataset' in execution_results['output_data']:
                dataset = execution_results['output_data']['dataset']
            elif 'dataset' in execution_results:
                dataset = execution_results['dataset']
        
        # Get data characteristics
        data_characteristics = state.get("data_characteristics", {})
        if not data_characteristics and 'data_characteristics' in state.get('execution_results', {}):
            data_characteristics = state['execution_results']['data_characteristics']
        
        return {
            "analysis_method": analysis_method,
            "dataset": dataset,
            "data_characteristics": data_characteristics,
            "validation_issues": state.get("data_quality_issues", [])
        }
    
    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract code blocks from markdown text."""
        # Pattern to match code blocks with optional language specifier
        pattern = r'```(?:\w*\n)?(.*?)```'
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return [block.strip() for block in code_blocks if block.strip()]
    
    async def __call__(self, state: AgentState) -> Command:
        """
        Generate analysis code based on the current state.
        
        Args:
            state: The current agent state
            
        Returns:
            Command: The next command to execute in the graph
        """
        print("--- [Chain] Analysis MAGI: 3. Generating Analysis Code ---")
        
        try:
            # Prepare context for code generation
            context = self._prepare_context(state)
            
            # If no analysis method is available, return an error
            if not context["analysis_method"] or context["analysis_method"] == "Perform a basic analysis.":
                error_msg = "No analysis method selected. Please run the method selector first."
                print(f"  > {error_msg}")
                return Command(
                    update={
                        "error": error_msg,
                        "messages": [{"role": "system", "content": error_msg}]
                    }
                )
            
            # Generate the analysis code
            result = await self.run(context, state.get("messages", []))
            
            # Prepare the code information for the state
            code_info = {
                "analysis_code": result.code,
                "code_description": result.description,
                "code_dependencies": result.dependencies,
                "code_parameters": result.parameters,
                "expected_outputs": result.expected_outputs,
                "messages": [{
                    "role": "system",
                    "content": f"Generated analysis code for: {context['analysis_method']}"
                }]
            }
            
            print(f"  > Generated analysis code with {len(result.dependencies)} dependencies")
            if result.expected_outputs:
                print(f"  > Expected outputs: {', '.join(result.expected_outputs)}")
            
            return Command(update=code_info)
            
        except Exception as e:
            error_msg = f"Error generating analysis code: {str(e)}"
            print(f"  > {error_msg}")
            return Command(
                update={
                    "error": error_msg,
                    "messages": [{"role": "system", "content": error_msg}]
                }
            )
    
    async def run(self, context: Dict[str, Any], messages: Optional[List[Dict[str, str]]] = None) -> AnalysisCode:
        """
        Generate analysis code based on the given context.
        
        Args:
            context: Dictionary containing analysis context
            messages: Previous conversation messages for context
            
        Returns:
            AnalysisCode: The generated analysis code and metadata
        """
        try:
            # Format the dataset for the prompt
            dataset_str = json.dumps(context["dataset"], indent=2) if isinstance(context["dataset"], (dict, list)) else str(context["dataset"])
            
            # Limit the size of the dataset in the prompt
            if len(dataset_str) > 2000:
                dataset_str = dataset_str[:1000] + "..." + dataset_str[-1000:]
            
            # Generate the code
            result = await self.chain.ainvoke({
                "analysis_method": context["analysis_method"],
                "dataset_json": dataset_str,
                "data_characteristics": json.dumps(context["data_characteristics"], indent=2),
                "validation_issues": json.dumps(context["validation_issues"], indent=2)
            })
            
            # Ensure the code has proper imports
            code = result.code
            if not any(imp in code for imp in ["import ", "from "]):
                # Add common imports if none are present
                imports = [
                    "import pandas as pd",
                    "import numpy as np",
                    "import matplotlib.pyplot as plt",
                    "import seaborn as sns",
                    "\n# Set plot style\nsns.set_style('whitegrid')\n"
                ]
                code = "\n".join(imports) + "\n\n" + code
            
            # Update the code in the result
            result.code = code
            
            return result
            
        except Exception as e:
            # Fallback to a simple analysis code if generation fails
            return AnalysisCode(
                code=(
                    "# Basic Data Analysis\n"
                    "import pandas as pd\n"
                    "import numpy as np\n"
                    "import matplotlib.pyplot as plt\n"
                    "import seaborn as sns\n\n"
                    "# Load and prepare data\n"
                    "df = pd.DataFrame(data)\n"
                    "print(\"\\nBasic Dataset Info:\")\n"
                    "print(df.info())\n"
                    "print(\"\\nDescriptive Statistics:\")\n"
                    "print(df.describe())\n"
                ),
                description="Basic data exploration with pandas",
                dependencies=["pandas", "numpy", "matplotlib", "seaborn"],
                parameters={"data": "The input dataset"},
                expected_outputs=["dataset_info", "descriptive_statistics"]
            )


# Create a singleton instance
analysis_code_generator_chain_instance = AnalysisCodeGeneratorChain()


async def analysis_code_generator_chain(state: AgentState) -> Command:
    """
    Entry point for the analysis code generator chain.
    
    Args:
        state: The current agent state
        
    Returns:
        Command: The next command to execute in the graph
    """
    return await analysis_code_generator_chain_instance(state)

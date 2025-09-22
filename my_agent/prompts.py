"""
Prompt templates for the MAGI research assistant system.

This module contains all the prompt templates used by different components of the MAGI system.
The prompts are organized into logical classes based on their purpose and the MAGI component they serve.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, Type, TypeVar, ClassVar, Optional
from pydantic import BaseModel

# Type variable for prompt classes
T = TypeVar('T', bound='BasePrompt')

class BasePrompt(BaseModel):
    """Base class for all prompt templates."""
    template: str
    description: str = ""
    
    @classmethod
    def get_prompts(cls: Type[T]) -> Dict[str, 'BasePrompt']:
        """Get all prompts defined in the class as a dictionary."""
        return {
            name: cls(template=prompt, description=getattr(cls, f"{name}_DESC", ""))
            for name, prompt in cls.__dict__.items()
            if not name.startswith('_') and isinstance(prompt, str) and '\n' in prompt
        }


from typing import ClassVar

class SurveyPrompts(BasePrompt):
    """Prompts for the Survey MAGI component."""
    
    TOPIC_CLARIFIER: ClassVar[str] = """
    You are a research assistant. Based on the user's initial research theme, 
    clarify and refine it into a concise, searchable topic.
    
    Initial Theme: {research_theme}
    
    Clarified Research Topic:
    """
    TOPIC_CLARIFIER_DESC: ClassVar[str] = "Refines a broad research theme into a focused, searchable topic."

    QUERY_GENERATOR: ClassVar[str] = """
    Generate 3-5 diverse and specific search queries for academic databases 
    based on the research theme.
    
    Research Theme: {research_theme}
    
    Search Queries:
    """
    QUERY_GENERATOR_DESC: ClassVar[str] = "Generates academic search queries from a research theme."

    RELEVANCE_FILTER: ClassVar[str] = """
あなたは優秀なリサーチアシスタントです。あなたの仕事は、与えられた調査テーマに基づき、検索結果のリストをフィルタリングし、関連性の高い文書だけを特定することです。

# 調査テーマ
{research_theme}

# 検索結果 (各結果には url, title, content が含まれます)
{search_results}

# 指示
1.  各文書が調査テーマにどれだけ関連しているかを評価してください。
2.  関連性が高いと判断した文書についてのみ、以下の情報を抽出・生成してください。
    - **url**: 元の検索結果から正確にコピーしてください。
    - **title**: 元の検索結果から正確にコピーしてください。
    - **summary**: なぜその文書が調査テーマと関連しているのか、理由を簡潔に要約してください。
    - **relevance_score**: 関連性を0から1のスコアで評価してください。
    - **content**: **【最重要】元の検索結果に含まれる`content`（本文）を、一切変更せずにそのまま含めてください。**
3.  最終的な出力は、必ず指定されたJSON形式に従ってください。

**出力形式の例:**
{{
  "relevant_documents": [
    {{
      "url": "（文書1のURL）",
      "title": "（文書1のタイトル）",
      "summary": "（この文書がなぜ関連しているかの要約）",
      "relevance_score": 0.9,
      "content": "（文書1の元のcontentをそのままここに記述）"
    }}
  ]
}}
"""
    RELEVANCE_FILTER_DESC: ClassVar[str] = "Filters and ranks search results by relevance."

    SUMMARY_GENERATOR: ClassVar[str] = """
あなたは優秀なリサーチアシスタントです。以下の調査テーマと、各文書の要約を基に、包括的な調査サマリーを作成してください。

# 調査テーマ
{research_theme}

# 参考文献リスト (各文書にはユニークな 'id' があります)
{relevant_docs}

# 各文書の詳細な要約 (各要約には対応する 'id' があります)
{individual_summaries}

# 指示
1.  `individual_summaries` の内容を統合し、調査テーマに関する**概要（overview）**を記述してください。
2.  `individual_summaries` から最も重要ないくつかの発見を**キーポイント（key_points）**としてリストアップしてください。
3.  **【最重要】** あなたが概要やキーポイントの根拠として利用した**要約（`individual_summaries`内）**を特定してください。
4.  その要約の `id` を使い、**`relevant_docs`リストから全く同じ`id`を持つ文書**を探し出してください。
5.  その文書の**タイトル（title）**と**URL（url）**を正確に抜き出し、**参考文献（references）**としてリストアップしてください。この対応付けは**絶対に間違えないでください。**
6.  最終的な出力は、必ず指定されたJSON形式に従ってください。
"""

    SUMMARY_GENERATOR_DESC: ClassVar[str] = "Creates a synthesized summary from multiple documents."

    HEARING_PROMPT: ClassVar[str] = """
You are an excellent research assistant. Your first job is to clarify the user's research goals through conversation.
Based on the conversation history below, determine if the user's research goal is clear.

## Judgment Criteria
- Does the user's input contain a specific research topic, keywords, or tasks?
  - **If YES:** Set "is_need_human_feedback" to `false`.
  - **If NO (e.g., it's just a greeting or an ambiguous question):** You MUST set "is_need_human_feedback" to `true` and generate an "additional_question" to ask for their goal.

## Examples
### Example 1
Conversation History:
human: こんにちは
Your response (JSON):
{{
  "is_need_human_feedback": true,
  "additional_question": "こんにちは！本日はどのような研究のお手伝いをしましょうか？"
}}

### Example 2
Conversation History:
human: LLMエージェントの評価方法について調べてほしい。
Your response (JSON):
{{
  "is_need_human_feedback": false,
  "additional_question": ""
}}

## Execute
Current Date: {current_date}
Conversation History:
{conversation_history}

Your response (JSON):
"""
    HEARING_PROMPT_DESC: ClassVar[str] = "Determines if additional information is needed from the user."


class PlanningPrompts(BasePrompt):
    """Prompts for the Planning MAGI component."""
    
    GOAL_SETTING: ClassVar[str] = """
    Define a clear and specific primary research goal based on the theme and survey.
    
    Research Theme: {research_theme}
    Survey Summary: {survey_summary}
    
    Primary Research Goal:
    """
    GOAL_SETTING_DESC: ClassVar[str] = "Defines a specific research goal from theme and survey."

    METHODOLOGY_SUGGESTER: ClassVar[str] = """
    Suggest a suitable methodology for the research goal.
    
    Research Goal: {research_goal}
    
    Suggested Methodology:
    """
    METHODOLOGY_SUGGESTER_DESC: ClassVar[str] = "Suggests appropriate research methodologies."

    EXPERIMENTAL_DESIGN: ClassVar[str] = """
    Design a detailed experiment based on the research goal and methodology.
    
    Research Goal: {research_goal}
    Methodology: {methodology}
    
    Experimental Design:
    """
    EXPERIMENTAL_DESIGN_DESC: ClassVar[str] = "Creates a detailed experimental design."

    TIMELINE_GENERATOR: ClassVar[str] = """
    Create a project timeline based on the experimental design.
    
    Experimental Design: {experimental_design}
    
    Project Timeline:
    """
    TIMELINE_GENERATOR_DESC: ClassVar[str] = "Generates a project timeline."


class ExecutionPrompts(BasePrompt):
    """Prompts for the Execution MAGI component."""
    
    PLAN_PARSER: ClassVar[str] = """
    Extract actionable tasks from the research plan.
    
    Research Plan (TeX):
    {research_plan_tex}
    
    Actionable Task List:
    """
    PLAN_PARSER_DESC: ClassVar[str] = "Extracts tasks from research plans."

    CODE_GENERATOR: ClassVar[str] = """
    Generate Python code for the specified tasks.
    
    Tasks to Execute:
    {tasks_to_execute}
    
    Python Code:
    """
    CODE_GENERATOR_DESC: ClassVar[str] = "Generates Python code for research tasks."


class AnalysisPrompts(BasePrompt):
    """Prompts for the Analysis MAGI component."""
    
    DATA_VALIDATOR: ClassVar[str] = """
    Validate the dataset and identify potential issues.
    
    Dataset Preview:
    {dataset_preview}
    
    Validation Report:
    """
    DATA_VALIDATOR_DESC: ClassVar[str] = "Validates dataset quality and completeness."

    METHOD_SELECTOR: ClassVar[str] = """
    Select the most appropriate analysis method.
    
    Research Goal: {research_goal}
    Dataset Preview: {dataset_preview}
    
    Recommended Analysis Method:
    """
    METHOD_SELECTOR_DESC: ClassVar[str] = "Recommends analysis methods."

    ANALYSIS_CODE_GENERATOR: ClassVar[str] = """
    Generate analysis code for the specified method and dataset.
    
    Analysis Method: {analysis_method}
    Dataset (JSON): {dataset_json}
    
    Analysis Code:
    """
    ANALYSIS_CODE_GENERATOR_DESC: ClassVar[str] = "Generates analysis code."

    RESULT_INTERPRETER: ClassVar[str] = """
    Interpret the analysis results in the context of the research goal.
    
    Research Goal: {research_goal}
    Analysis Results: {analysis_results}
    
    Interpretation:
    """
    RESULT_INTERPRETER_DESC: ClassVar[str] = "Interprets analysis results."

    CONCLUSION_GENERATOR: ClassVar[str] = """
    Formulate research conclusions based on the interpretation.
    
    Research Goal: {research_goal}
    Interpretation: {interpretation}
    
    Conclusion:
    """
    CONCLUSION_GENERATOR_DESC: ClassVar[str] = "Generates research conclusions."


class ReportPrompts(BasePrompt):
    """Prompts for the Report MAGI component."""
    
    STRUCTURE_PLANNER: ClassVar[str] = """
    Plan the structure of the research paper.
    
    Research Goal: {research_goal}
    
    Paper Structure:
    """
    STRUCTURE_PLANNER_DESC: ClassVar[str] = "Plans paper structure."

    SECTION_WRITER: ClassVar[str] = """
    Write the content for a research paper section.
    
    Paper Structure: {paper_structure}
    Content: {aggregated_content}
    
    Section Content:
    """
    SECTION_WRITER_DESC: ClassVar[str] = "Writes paper sections."

    TEX_COMPILER: ClassVar[str] = """
    Convert draft content into a complete LaTeX document.
    
    Title: {paper_title}
    Content: {draft_content}
    
    LaTeX Document:
    """
    TEX_COMPILER_DESC: ClassVar[str] = "Generates LaTeX documents."

    FINAL_REVIEWER: ClassVar[str] = """
    Review and polish the LaTeX document.
    
    Document to Review:
    ```latex
    {tex_document}
    ```
    
    Reviewed Document:
    """
    FINAL_REVIEWER_DESC: ClassVar[str] = "Reviews and polishes documents."


def load_prompt(name: str, base_path: Optional[Path] = None) -> str:
    """
    Load a prompt from a file.
    
    Args:
        name: Name of the prompt file (without .prompt extension)
        base_path: Base directory containing the prompts directory. If None, looks in the current directory.
        
    Returns:
        str: The content of the prompt file
    """
    # Convert base_path to Path if it's a string
    if base_path is not None and not isinstance(base_path, Path):
        base_path = Path(base_path)
    
    # Look in the current directory by default
    if base_path is None:
        base_path = Path.cwd()
    
    # First try the exact path if it's a full path
    if Path(name).is_absolute():
        return Path(name).read_text(encoding="utf-8").strip()
    
    # Try to find the prompt file in the base_path/prompts directory
    prompt_path = base_path / "prompts" / f"{name}.prompt"
    
    if not prompt_path.exists():
        # Try one level up if not found
        prompt_path = base_path.parent / "prompts" / f"{name}.prompt"
    
    if not prompt_path.exists():
        # If still not found, try in the same directory as prompts.py
        prompt_path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find prompt file: {name}.prompt in any of the searched locations")
    
    return prompt_path.read_text(encoding="utf-8").strip()


class PromptManager:
    """Manages and provides access to all prompts."""
    
    PROMPT_CLASSES = {
        'survey': SurveyPrompts,
        'planning': PlanningPrompts,
        'execution': ExecutionPrompts,
        'analysis': AnalysisPrompts,
        'report': ReportPrompts
    }
    
    @classmethod
    def get_prompt(cls, component: str, prompt_name: str) -> str:
        """Get a specific prompt template string by component and name."""
        prompt_class = cls.PROMPT_CLASSES.get(component.lower())
        if not prompt_class:
            raise ValueError(f"Invalid component: {component}")
            
        prompt = getattr(prompt_class, prompt_name.upper(), None)
        if not prompt:
            raise ValueError(f"Invalid prompt name: {prompt_name}")
            
        # If it's a BasePrompt instance, get the template
        if isinstance(prompt, BasePrompt):
            return prompt.template
            
        # If it's a string, return it directly
        if isinstance(prompt, str):
            return prompt
            
        return BasePrompt(
            template=prompt,
            description=getattr(prompt_class, f"{prompt_name.upper()}_DESC", "")
        )
    
    @classmethod
    def get_all_prompts(cls) -> Dict[str, Dict[str, BasePrompt]]:
        """Get all prompts organized by component."""
        return {
            name: prompt_class.get_prompts()
            for name, prompt_class in cls.PROMPT_CLASSES.items()
        }


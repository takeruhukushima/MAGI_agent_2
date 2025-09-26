import json
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 親グラフのState定義をインポートするのがベストプラクティスです
# from my_agent.science_magi import ScienceAgentState
# 見つからない場合は一時的にdictとして扱います
try:
    from my_agent.science_magi import ScienceAgentState
except ImportError:
    ScienceAgentState = dict

from my_agent.prompts import PromptManager
from my_agent.settings import settings


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
        prompt_template = PromptManager.get_prompt('report', 'TEX_COMPILER')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # このChainは生のTeX文字列を生成するため、with_structured_outputは不要
        self.chain = self.prompt | self.llm

    def _get_planning_elements(self, state: ScienceAgentState) -> Dict[str, str]:
        """Extract and format planning elements from the state for the prompt."""
        research_goal = state.get('research_goal_result', {})
        return {
            "paper_title": f"Research Plan: {research_goal.get('title', 'Untitled Research')}",
            "draft_content": self._format_draft_content(state)
        }
    
    def _format_draft_content(self, state: ScienceAgentState) -> str:
        """状態から人間が読めるレポート用コンテンツを作成"""
        research_goal = state.get("research_goal_result", {})
        experimental_design = state.get("experimental_design_result", {})
        timeline = state.get("timeline_result", {})
        methodology = state.get("methodology_result", {})
        
        # 各辞書から必要な値を抽出し、人間が読めるテキストを組み立てる
        # json.dumpsではなく、値を取り出して整形する
        content = f"""
\\section*{{Research Goal}}
\\subsection*{{{research_goal.get('title', 'N/A')}}}
{research_goal.get('description', 'No description.')}

\\subsection*{{Objectives}}
\\begin{{itemize}}
{"".join([f"  \\item {obj}\\n" for obj in research_goal.get('objectives', [])])}
\\end{{itemize}}

\\section*{{Methodology}}
\\subsection*{{{methodology.get('name', 'N/A')}}}
{methodology.get('description', 'No description.')}

\\section*{{Experimental Design}}
\\subsection*{{{experimental_design.get('title', 'N/A')}}}
\\subsubsection*{{Hypothesis}}
{experimental_design.get('hypothesis', 'N/A')}
\\subsubsection*{{Variables}}
{experimental_design.get('variables_description', 'N/A')}
\\subsubsection*{{Timeline}}
{timeline.get('timeline_description', 'N/A')}
"""
        return content

    def __call__(self, state: ScienceAgentState) -> Dict[str, Any]:
        """langgraphのノードとして呼び出されるメソッド"""
        print("--- [Chain] Planning MAGI: 5. Formatting TeX Report ---")
        try:
            tex_document = self.run(state)
            if tex_document and "% Error" not in tex_document:
                print("  > TeX content generated successfully.")
                return {"tex_document": tex_document}
            else:
                error_msg = "Failed to generate TeX content."
                print(f"  > {error_msg}")
                return {"tex_document": "% Failed to generate TeX report.", "error": error_msg}
        except Exception as e:
            error_msg = f"Error formatting TeX: {str(e)}"
            print(f"  > {error_msg}")
            return {"tex_document": f"% Error: {str(e)}", "error": error_msg}
    
    def run(self, state: ScienceAgentState) -> str:
        """すべての計画要素をTeX文書に整形する"""
        try:
            if not state:
                raise ValueError("No state provided for TeX formatting")
            
            formatted_input = self._get_planning_elements(state)
            # LLMを呼び出し、AIMessageオブジェクトを取得
            response = self.chain.invoke(formatted_input)
            
            # AIMessageオブジェクトから生のテキストコンテンツを返す
            return response.content
            
        except Exception as e:
            return (
                "% Error generating TeX document\n"
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                f"% Error: {str(e)}\n"
                "Failed to generate TeX document.\n"
                "\\end{document}"
            )
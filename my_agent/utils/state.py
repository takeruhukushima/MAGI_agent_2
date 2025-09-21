from typing import TypedDict, Dict, Any, List, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

# total=False を設定し、キーが任意であることを示します
class AgentState(TypedDict, total=False):
    """
    MAGIシステム全体で共有される状態（プロジェクトの共有フォルダ）。
    """
    # --- プロセスの進行状況を管理 ---
    # 最後に完了したタスクの名前 (survey, planning, analysis) を記録する
    last_completed_task: str

    # --- プロジェクトのコア情報 ---
    research_theme: str
    
    # --- Survey MAGIの成果物 ---
    clarified_theme: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    relevant_docs: List[Dict[str, Any]]
    survey_summary: str
    
    # --- Planning MAGIの成果物 ---
    research_goal: str
    methodology: str
    experimental_design: str
    timeline: str
    research_plan_tex: str
    
    # --- Execution MAGIの成果物 (今回は使いませんが定義は残します) ---
    tasks_to_execute: List[str]
    generated_code: str
    execution_log: str
    simulation_output: Dict[str, Any]
    execution_results: Dict[str, Any]
    
    # --- Analysis MAGIの成果物 ---
    is_data_valid: bool
    data_validation_summary: str
    analysis_method: str
    analysis_code: str
    analysis_output: Dict[str, Any]
    result_interpretation: str
    analysis_results: Dict[str, Any]
    
    # --- Report MAGIの成果物 (今回は使いませんが定義は残します) ---
    paper_structure: List[str]
    aggregated_content: str
    draft_content: str
    compiled_tex: str
    final_report_tex: str
    
    # --- 会話とフィードバックの履歴 ---
    messages: Annotated[Sequence[BaseMessage], add_messages]


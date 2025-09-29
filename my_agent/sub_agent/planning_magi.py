import asyncio
import operator
from typing import Annotated, TypedDict, Literal, Dict, Any
from typing_extensions import TypedDict
from langgraph.types import Command, interrupt

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

# from arxiv_researcher.agent.paper_analyzer_agent import (
#     PaperAnalyzerAgent,
#     PaperAnalyzerAgentInputState,
# )
from my_agent.chains.planning_magi.goal_setting_chain import GoalSettingChain
from my_agent.chains.planning_magi.experimental_design_chain import ExperimentalDesignChain
from my_agent.chains.planning_magi.timeline_generator_chain import TimelineGeneratorChain
from my_agent.chains.planning_magi.methodology_suggester_chain import MethodologySuggesterChain
from my_agent.chains.planning_magi.tex_formatter_chain import TexFormatterChain

from my_agent.models import ReadingResult
# from my_agent.searcher.arxiv_searcher import ArxivSearcher
from my_agent.settings import settings

# from my_agent.sub_agent.survey_magi import SurveyAgentState # SurveyAgentStateをインポート

class PlanningAgentInputState(TypedDict):
    # Survey Agentからの入力を受け取れるように拡張
    research_theme: str
    decomposed_tasks: list[str]
    # 必要であれば、SurveyAgentの他の出力もここに追加
    survey_summary: dict

class PlanningAgentProcessState(TypedDict):
    # 各ステップの出力を保存
    research_goal_result: dict
    experimental_design_result: dict
    timeline_result: dict
    methodology_result: dict
    tex_document: str
 

class PlanningAgentOutputState(TypedDict):
    reading_results: list[ReadingResult]


class PlanningAgentState(
    PlanningAgentInputState,
    PlanningAgentProcessState,
    PlanningAgentOutputState,
):
    pass


class PlanningAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.recursion_limit = settings.langgraph.max_recursion_limit
        # self.max_workers = settings.arxiv_search_agent.max_workers
        self.llm = llm
        # self.searcher = searcher
        # self.paper_processor = PaperProcessor(
        #     searcher=self.searcher, max_workers=self.max_workers
        # )
        self.goal_setting = GoalSettingChain(llm)
        self.experimental_design = ExperimentalDesignChain(llm)
        self.timeline_generator = TimelineGeneratorChain(llm)
        self.methodology_suggester = MethodologySuggesterChain(llm)
        self.tex_formatter = TexFormatterChain(llm)
        
        self.graph = self._create_graph()

    def __call__(self, state: PlanningAgentInputState) -> CompiledStateGraph:
        if state is None:
            return self.graph
            
        # ▼▼▼ Survey Agentからのデータ変換を修正 ▼▼▼
        processed_state = state.copy()
        
        # research_themeが空または空文字列の場合、survey_summaryから抽出
        research_theme = processed_state.get('research_theme', '').strip()
        if not research_theme and 'survey_summary' in processed_state:
            survey_summary = processed_state['survey_summary']
            if isinstance(survey_summary, dict) and 'overview' in survey_summary:
                # overviewから研究テーマを抽出（最初の文を使用）
                overview = survey_summary['overview']
                first_sentence = overview.split('.')[0] + '.'
                processed_state['research_theme'] = first_sentence
            elif isinstance(survey_summary, dict) and 'key_points' in survey_summary:
                # key_pointsがある場合は最初のポイントを使用
                key_points = survey_summary['key_points']
                if key_points and isinstance(key_points, list):
                    processed_state['research_theme'] = key_points[0]
        
        # 後方互換性のため、goalもresearch_themeから設定
        if 'research_theme' in processed_state and 'goal' not in processed_state:
            processed_state['goal'] = processed_state['research_theme']
        
        # decomposed_tasksが空の場合のデフォルト値設定
        if not processed_state.get('decomposed_tasks'):
            processed_state['decomposed_tasks'] = ["研究計画を立案する"]
            
        # tasksもdecomposed_tasksから設定（後方互換性）
        if 'decomposed_tasks' in processed_state and 'tasks' not in processed_state:
            processed_state['tasks'] = processed_state['decomposed_tasks']
        
        print("=== Planning Agent State Debug ===")
        print(f"Original research_theme: '{state.get('research_theme', 'Not found')}'")
        print(f"Processed research_theme: '{processed_state.get('research_theme', 'Not found')}'")
        print(f"Survey summary exists: {'survey_summary' in processed_state}")
        print("=================================")
         
        # 同期実行に変更（全てのチェーンが同期なので）
        return self.graph.invoke(processed_state)
 
    async def async_invoke(self, state):
        """非同期版の呼び出し（必要に応じて）"""
        # 同じ処理を非同期版でも適用
        processed_state = state.copy()
        
        # research_themeが空または空文字列の場合、survey_summaryから抽出
        research_theme = processed_state.get('research_theme', '').strip()
        if not research_theme and 'survey_summary' in processed_state:
            survey_summary = processed_state['survey_summary']
            if isinstance(survey_summary, dict) and 'overview' in survey_summary:
                overview = survey_summary['overview']
                first_sentence = overview.split('.')[0] + '.'
                processed_state['research_theme'] = first_sentence
            elif isinstance(survey_summary, dict) and 'key_points' in survey_summary:
                key_points = survey_summary['key_points']
                if key_points and isinstance(key_points, list):
                    processed_state['research_theme'] = key_points[0]
        
        if 'research_theme' in processed_state and 'goal' not in processed_state:
            processed_state['goal'] = processed_state['research_theme']
            
        if not processed_state.get('decomposed_tasks'):
            processed_state['decomposed_tasks'] = ["研究計画を立案する"]
            
        if 'decomposed_tasks' in processed_state and 'tasks' not in processed_state:
            processed_state['tasks'] = processed_state['decomposed_tasks']
            
        return await self.graph.ainvoke(processed_state)

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            PlanningAgentState,
            input=PlanningAgentInputState,
            output=PlanningAgentOutputState,
        )
        workflow.add_node("goal_setting", self.goal_setting)
        workflow.add_node("methodology_suggester", self.methodology_suggester)
        workflow.add_node("experimental_design", self.experimental_design)
        workflow.add_node("timeline_generator", self.timeline_generator)
        workflow.add_node("tex_formatter", self.tex_formatter)
        workflow.add_node("final_output", self._final_output_handler)

        workflow.add_edge("tex_formatter", "final_output")

        workflow.set_entry_point("goal_setting")
        workflow.set_finish_point("final_output")

        return workflow.compile()

    def _final_output_handler(self, state: PlanningAgentState) -> dict:
        """最終出力をまとめ、ユーザーの承認を待つ"""
        tex_document = state.get("tex_document", "")
        print("\n--- DEBUG: Final Output Handler Inputs ---")
        print(f"Type of tex_document: {type(tex_document)}")
        print(f"Length of tex_document: {len(tex_document)}")
        print("Value of tex_document:")
        print(f"'''{tex_document}'''") # 中身が分かりやすいようにクォートで囲む
        print("----------------------------------------\n")
        if tex_document and "% Error" not in tex_document:
            research_goal = state.get("research_goal_result", {})
            experimental_design = state.get("experimental_design_result", {})
            timeline = state.get("timeline_result", {})
            methodology = state.get("methodology_result", {})

            final_message = f"""Planning MAGIによる実験計画が完成しました。

## 生成要素:
- 研究目標: {research_goal.get('title', 'N/A')}
- 推奨方法論: {methodology.get('name', 'N/A')}
- 実験設計の概要: {experimental_design.get('objective', 'N/A')}
- タイムラインの概要: {timeline.get('project_name', 'N/A')}

## TeX形式の研究計画書:
```tex
{tex_document}
```
この計画で次のステップに進みますか？
承認する場合は、Enterキーを押すか、「はい」などと入力して続行してください。
"""
            interrupt(final_message)    
            return {"final_output": final_message}
        else:
            error_msg = "実験計画の生成に失敗しました。"
            interrupt(error_msg)
            return {"final_output": error_msg}

     # def _analyze_paper(self, state: PaperAnalyzerAgentInputState) -> dict:

    # def _analyze_paper(self, state: PaperAnalyzerAgentInputState) -> dict:
    #     output = self.paper_analyzer.graph.invoke(
    #         state,
    #         config={
    #             "recursion_limit": self.recursion_limit,
    #         },
    #     )
    #     reading_result = output.get("reading_result")
    #     return {
    #         "processing_reading_results": [reading_result] if reading_result else []
    #     }

    # def _organize_results(self, state: PlanningAgentState) -> dict:
    #     processing_reading_results = state.get("processing_reading_results", [])
    #     reading_results = []

    #     # 関連性のある論文のみをフィルタリング
    #     for result in processing_reading_results:
    #         if result and result.is_related:
    #             reading_results.append(result)

    #     return {"reading_results": reading_results}

    def _human_feedback(
        self, state: PlanningAgentState
    ) -> Command[Literal["user_hearing"]]:
        # 最後のメッセージを取得
        last_message = state["messages"][-1]
        # ユーザーへの質問を表示
        human_feedback = interrupt(last_message.content)
        if human_feedback is None:
            human_feedback = "そのままの条件で検索し、調査してください。"
        return Command(
            goto="user_hearing",
            update={"messages": [{"role": "human", "content": human_feedback}]},
        )


graph = PlanningAgent(
    settings.fast_llm,
).graph

if __name__ == "__main__":
    agent = PlanningAgent(settings.fast_llm)
    initial_state: PlanningAgentState = {
        "goal": "LLMエージェントの評価方法について調べる",
        "tasks": [
            "2023年以降に発表された論文をarXivから調査し、最新のLLMエージェント評価用データセットを収集する。"
        ],
    }

    for event in agent.graph.stream(
        input=initial_state,
        config={"recursion_limit": settings.langgraph.max_recursion_limit},
        stream_mode="updates",
        subgraphs=True,
    ):
        parent, update_state = event

        # 実行ノードの情報を取得
        node = list(update_state.keys())[0]

        # parentが空の()でない場合
        if parent:
            print(f"{parent}: {node}")
        else:
            print(node)
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


class PlanningAgentInputState(TypedDict):
    goal: str
    tasks: list[str]


class PlanningAgentProcessState(TypedDict):
    processing_reading_results: Annotated[list[ReadingResult], operator.add]


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

    def __call__(self, state=None) -> CompiledStateGraph:
        if state is None:
            return self.graph
        return asyncio.run(self.graph.ainvoke(state))  # asyncio.runでラップ

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            PlanningAgentState,
            input=PlanningAgentInputState,
            output=PlanningAgentOutputState,
        )
        workflow.add_node("goal_setting", self.goal_setting)
        workflow.add_node("experimental_design", self.experimental_design)
        workflow.add_node("timeline_generator", self.timeline_generator)
        workflow.add_node("methodology_suggester", self.methodology_suggester)
        workflow.add_node("tex_formatter", self.tex_formatter)
        
        workflow.set_entry_point("goal_setting")
        workflow.set_finish_point("tex_formatter")

        return workflow.compile()

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

    # def _organize_results(self, state: PaperSearchAgentState) -> dict:
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

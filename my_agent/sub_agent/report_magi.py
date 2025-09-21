import operator
from typing import Annotated, TypedDict,Literal
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

from my_agent.chains.report_magi.structure_planner_chain import StructurePlannerChain
from my_agent.chains.report_magi.content_aggregator_chain import ContentAggregatorChain
from my_agent.chains.report_magi.final_review_chain import FinalReviewChain
from my_agent.chains.report_magi.section_writer_chain import SectionWriterChain
from my_agent.chains.report_magi.tex_generator_chain import TexGeneratorChain

from my_agent.models import ReadingResult
# from my_agent.searcher.arxiv_searcher import ArxivSearcher
from my_agent.settings import settings


class ReportAgentInputState(TypedDict):
    goal: str
    tasks: list[str]


class ReportAgentProcessState(TypedDict):
    processing_reading_results: Annotated[list[ReadingResult], operator.add]


class ReportAgentOutputState(TypedDict):
    reading_results: list[ReadingResult]


class ReportAgentState(
    ReportAgentInputState,
    ReportAgentProcessState,
    ReportAgentOutputState,
):
    pass


class ReportAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.recursion_limit = settings.langgraph.max_recursion_limit
        # self.max_workers = settings.arxiv_search_agent.max_workers
        self.llm = llm
        # self.searcher = searcher
        # self.paper_processor = PaperProcessor(
        #     searcher=self.searcher, max_workers=self.max_workers
        # )
        self.structure_planner = StructurePlannerChain(llm)
        self.content_aggregator = ContentAggregatorChain(llm)
        self.section_writer = SectionWriterChain(llm)
        self.final_review = FinalReviewChain(llm)
        self.tex_generator = TexGeneratorChain(llm)
        
        self.graph = self._create_graph()

    def __call__(self, state=None) -> CompiledStateGraph:
        if state is None:
            return self.graph
        return self.graph.invoke(state)

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            ReportAgentState,
            input=ReportAgentInputState,
            output=ReportAgentOutputState,
        )
        # Add all nodes
        workflow.add_node("structure_planner", self.structure_planner)
        workflow.add_node("section_writer", self.section_writer)
        workflow.add_node("content_aggregator", self.content_aggregator)
        workflow.add_node("final_review", self.final_review)
        workflow.add_node("tex_generator", self.tex_generator)
        
        # Define the workflow edges
        workflow.add_edge("structure_planner", "section_writer")
        
        # Conditional edge for section writer - either go to next section or to content_aggregator
        def route_after_section(state: dict) -> str:
            paper_structure = state.get("paper_structure", {})
            sections = paper_structure.get("sections", [])
            current_idx = state.get("current_section_index", 0)
            
            if current_idx < len(sections):
                return "section_writer"  # More sections to write
            return "content_aggregator"  # All sections written
            
        workflow.add_conditional_edges(
            "section_writer",
            route_after_section,
            {
                "section_writer": "section_writer",
                "content_aggregator": "content_aggregator"
            }
        )
        
        # Add remaining edges
        workflow.add_edge("content_aggregator", "final_review")
        workflow.add_edge("final_review", "tex_generator")
        
        # Set entry and finish points
        workflow.set_entry_point("structure_planner")
        workflow.set_finish_point("tex_generator")

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
        self, state: ReportAgentState
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


graph = ReportAgent(
    settings.fast_llm,
).graph

if __name__ == "__main__":
    agent = ReportAgent(settings.fast_llm)
    initial_state: ReportAgentState = {
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

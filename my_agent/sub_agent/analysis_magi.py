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
from my_agent.chains.analysis_magi.analysis_code_generator_chain import AnalysisCodeGeneratorChain
from my_agent.chains.analysis_magi.data_validator_chain import DataValidatorChain
from my_agent.chains.analysis_magi.method_selector_chain import MethodSelectorChain
from my_agent.chains.analysis_magi.result_interpreter_chain import ResultInterpreterChain
from my_agent.chains.analysis_magi.conclusion_generator_chain import ConclusionGeneratorChain

from my_agent.models import ReadingResult
# from my_agent.searcher.arxiv_searcher import ArxivSearcher
from my_agent.settings import settings


class AnalysisAgentInputState(TypedDict):
    goal: str
    tasks: list[str]


class AnalysisAgentProcessState(TypedDict):
    processing_reading_results: Annotated[list[ReadingResult], operator.add]


class AnalysisAgentOutputState(TypedDict):
    reading_results: list[ReadingResult]


class AnalysisAgentState(
    AnalysisAgentInputState,
    AnalysisAgentProcessState,
    AnalysisAgentOutputState,
):
    pass


class AnalysisAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.recursion_limit = settings.langgraph.max_recursion_limit
        # self.max_workers = settings.arxiv_search_agent.max_workers
        self.llm = llm
        # self.searcher = searcher
        # self.paper_processor = PaperProcessor(
        #     searcher=self.searcher, max_workers=self.max_workers
        # )
        self.analysis_code_generator = AnalysisCodeGeneratorChain(llm)
        self.data_validator = DataValidatorChain(llm)
        self.method_selector = MethodSelectorChain(llm)
        self.result_interpreter = ResultInterpreterChain(llm)
        self.conclusion_generator = ConclusionGeneratorChain(llm)
        
        self.graph = self._create_graph()

    def __call__(self, state=None) -> CompiledStateGraph:
        if state is None:
            return self.graph
        return self.graph.invoke(state)

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            AnalysisAgentState,
            input=AnalysisAgentInputState,
            output=AnalysisAgentOutputState,
        )
        workflow.add_node("analysis_code_generator", self.analysis_code_generator)
        workflow.add_node("data_validator", self.data_validator)
        workflow.add_node("method_selector", self.method_selector)
        workflow.add_node("result_interpreter", self.result_interpreter)
        workflow.add_node("conclusion_generator", self.conclusion_generator)
        
        workflow.set_entry_point("analysis_code_generator")
        workflow.set_finish_point("conclusion_generator")

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
        self, state: AnalysisAgentState
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


graph = AnalysisAgent(
    settings.fast_llm,
).graph

if __name__ == "__main__":
    agent = AnalysisAgent(settings.fast_llm)
    initial_state: AnalysisAgentState = {
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

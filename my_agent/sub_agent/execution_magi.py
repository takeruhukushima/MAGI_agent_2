# 一旦忘れる。

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
from my_agent.chains.execution_magi.code_generator_chain import CodeGeneratorChain
from my_agent.chains.execution_magi.data_logger_chain import DataLoggerChain
from my_agent.chains.execution_magi.human_approval_chain import HumanApprovalChain
from my_agent.chains.execution_magi.plan_parser_chain import PlanParserChain
from my_agent.chains.execution_magi.simulation_executor_chain import SimulationExecutorChain

from my_agent.models import ReadingResult
# from my_agent.searcher.arxiv_searcher import ArxivSearcher
from my_agent.settings import settings


class ExecutionAgentInputState(TypedDict):
    goal: str
    tasks: list[str]


class ExecutionAgentProcessState(TypedDict):
    processing_reading_results: Annotated[list[ReadingResult], operator.add]


class ExecutionAgentOutputState(TypedDict):
    reading_results: list[ReadingResult]


class ExecutionAgentState(
    ExecutionAgentInputState,
    ExecutionAgentProcessState,
    ExecutionAgentOutputState,
):
    pass


class ExecutionAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.recursion_limit = settings.langgraph.max_recursion_limit
        # self.max_workers = settings.arxiv_search_agent.max_workers
        self.llm = llm
        # self.searcher = searcher
        # self.paper_processor = PaperProcessor(
        #     searcher=self.searcher, max_workers=self.max_workers
        # )
        # Initialize chains with correct parameters
        self.code_generator = CodeGeneratorChain(llm)  # Needs LLM
        self.data_logger = DataLoggerChain()  # No parameters needed
        self.human_approval = HumanApprovalChain()  # No parameters needed
        self.plan_parser = PlanParserChain(llm)  # Needs LLM
        self.simulation_executor = SimulationExecutorChain()  # No parameters needed
        
        self.graph = self._create_graph()

    def __call__(self, state=None) -> CompiledStateGraph:
        if state is None:
            return self.graph
        return self.graph.invoke(state)

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            ExecutionAgentState,
            input=ExecutionAgentInputState,
            output=ExecutionAgentOutputState,
        )
        workflow.add_node("code_generator", self.code_generator)
        workflow.add_node("data_logger", self.data_logger)
        workflow.add_node("human_approval", self.human_approval)
        workflow.add_node("plan_parser", self.plan_parser)
        workflow.add_node("simulation_executor", self.simulation_executor)
        
        workflow.set_entry_point("code_generator")
        workflow.set_finish_point("simulation_executor")

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
        self, state: ExecutionAgentState
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


graph = ExecutionAgent(
    settings.fast_llm,
).graph

if __name__ == "__main__":
    agent = ExecutionAgent(settings.fast_llm)
    initial_state: ExecutionAgentState = {
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

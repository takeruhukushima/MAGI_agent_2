from typing import Annotated, Literal, TypedDict

from my_agent.settings import settings

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from my_agent.sub_agent.survey_magi import SurveyAgent
from my_agent.sub_agent.planning_magi import PlanningAgent
from my_agent.sub_agent.execution_magi import ExecutionAgent
from my_agent.sub_agent.analysis_magi import AnalysisAgent
from my_agent.sub_agent.report_magi import ReportAgent

class ScienceAgentInputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ScienceAgentPrivateState(TypedDict):
    survey_agent: SurveyAgent
    planning_agent: PlanningAgent
    execution_agent: ExecutionAgent
    analysis_agent: AnalysisAgent
    report_agent: ReportAgent

    research_theme: str  # ← これが必要
    survey_summary: dict  # ← これが必要

    final_output: str
    retry_count: int


class ScienceAgentOutputState(TypedDict):
    final_output: str


class ScienceAgentState(
    ScienceAgentInputState, ScienceAgentPrivateState, ScienceAgentOutputState
):
    pass


class ScienceAgent:
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI = settings.llm,
        fast_llm: ChatGoogleGenerativeAI = settings.llm,
        reporter_llm: ChatGoogleGenerativeAI = settings.llm,
    ) -> None:
        self.recursion_limit = settings.langgraph.max_recursion_limit
        self.max_evaluation_retry_count = (
            settings.science_agent.max_evaluation_retry_count
        )
        self.survey_agent = SurveyAgent(llm)
        self.planning_agent = PlanningAgent(llm)
        self.execution_agent = ExecutionAgent(llm)
        self.analysis_agent = AnalysisAgent(llm)
        self.report_agent = ReportAgent(llm)
        self.graph = self._create_graph()
    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            state_schema=ScienceAgentState,
            input=ScienceAgentInputState,
            output=ScienceAgentOutputState,
        )
        # Use distinct node names that don't conflict with state keys
        workflow.add_node("survey", self.survey_agent)
        workflow.add_node("planning", self.planning_agent)
        workflow.add_node("execution", self.execution_agent)  # Uncomment when ready
        workflow.add_node("analysis", self.analysis_agent)
        workflow.add_node("generate_report", self.report_agent)
        
        # Define the workflow edges
        workflow.add_edge("survey", "planning")
        workflow.add_edge("planning", "analysis")
        workflow.add_edge("planning", "execution")  # Uncomment when ready
        workflow.add_edge("execution", "analysis")  # Uncomment when ready
        workflow.add_edge("analysis", "generate_report")
        
        workflow.set_entry_point("survey")
        workflow.set_finish_point("generate_report")

        return workflow.compile()

    # def _human_feedback(
    #     self, state: ResearchAgentState
    # ) -> Command[Literal["user_hearing"]]:
    #     # 最後のメッセージを取得
    #     last_message = state["messages"][-1]
    #     # ユーザーへの質問を表示
    #     human_feedback = interrupt(last_message.content)
    #     if human_feedback is None:
    #         human_feedback = "そのままの条件で検索し、調査してください。"
    #     return Command(
    #         goto="user_hearing",
    #         update={"messages": [{"role": "human", "content": human_feedback}]},
    #     )

    # def _paper_search_agent(
    #     self, state: ResearchAgentState
    # ) -> Command[Literal["evaluate_task"]]:
    #     output = self.paper_search_agent.graph.invoke(
    #         input=state,
    #         config={"recursion_limit": settings.langgraph.max_recursion_limit},
    #     )
    #     return Command(
    #         goto="evaluate_task",
    #         update={"reading_results": output.get("reading_results", [])},
    #     )


graph = ScienceAgent().graph

if __name__ == "__main__":
    png_file = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_file)

import operator
from typing import List,Dict, Any,Annotated, TypedDict,Literal
from typing_extensions import TypedDict
from langgraph.types import Command, interrupt

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import AnyMessage, add_messages
# from arxiv_researcher.agent.paper_analyzer_agent import (
#     PaperAnalyzerAgent,
#     PaperAnalyzerAgentInputState,
# )
from my_agent.chains.survey_magi.hearing_chain import HearingChain
from my_agent.chains.survey_magi.topic_clarification_chain import TopicClarificationChain
from my_agent.chains.survey_magi.search_executor_chain import SearchExecutorChain
from my_agent.chains.survey_magi.relevance_filter_chain import RelevanceFilterChain
# from my_agent.chains.survey_magi.web_page_reader_chain import WebPageReaderChain
from my_agent.chains.survey_magi.summary_generator_chain import SummaryGeneratorChain
from my_agent.chains.survey_magi.query_generator_chain import QueryGeneratorChain
from my_agent.chains.survey_magi.document_summarizer_chain import DocumentSummarizerChain


from my_agent.models import ReadingResult
from my_agent.settings import settings


class SurveyAgentInputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class SurveyAgentProcessState(TypedDict):
    processing_reading_results: Annotated[list[ReadingResult], operator.add]
    hearing: HearingChain | None
    
    # ▼▼▼▼▼▼▼ 以下を追加 ▼▼▼▼▼▼▼
    research_theme: str | None          # 明確化された研究テーマ
    search_queries: list[str] | None    # 生成された検索クエリのリスト
    search_results: list[dict] | None # 検索エンジンからの生の結果
    relevant_documents: List[dict] | None
    individual_summaries: List[dict] | None # ← この行を追加
    # ▲▲▲▲▲▲▲ ここまでを追加 ▲▲▲▲▲▲▲


class SurveyAgentOutputState(TypedDict):
    survey_summary: Any | None
    status: str | None # 処理が成功したか、エラーになったかを示す


class SurveyAgentState(
    SurveyAgentInputState,
    SurveyAgentProcessState,
    SurveyAgentOutputState,
):
    pass


class SurveyAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.recursion_limit = settings.langgraph.max_recursion_limit
        # self.max_workers = settings.arxiv_search_agent.max_workers
        self.llm = llm
        # self.searcher = searcher
        # self.paper_processor = PaperProcessor(
        #     searcher=self.searcher, max_workers=self.max_workers
        # )
        self.user_hearing = HearingChain(llm)
        self.topic_clarification = TopicClarificationChain(llm)
        self.query_generator = QueryGeneratorChain(llm)
        self.search_executor = SearchExecutorChain(llm)
        self.relevance_filter = RelevanceFilterChain(llm)
        # self.web_page_reader = WebPageReaderChain(llm)
        self.document_summarizer = DocumentSummarizerChain(llm)
        self.summary_generator = SummaryGeneratorChain(llm)
        
        self.graph = self._create_graph()

    def __call__(self, state=None) -> CompiledStateGraph:
        if state is None:
            return self.graph
        return self.graph.invoke(state)

    def _should_request_feedback(self, state: dict) -> Literal["human_feedback", "topic_clarification"]:
        """stateからhearingオブジェクトを安全に取得し、分岐先を決定する"""
        hearing_result = state.get("hearing")
        if hearing_result and hearing_result.is_need_human_feedback:
            return "human_feedback"
        return "topic_clarification"

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            SurveyAgentState,
            input=SurveyAgentInputState,
            output=SurveyAgentOutputState,
        )
        # ノードの追加
        workflow.add_node("user_hearing", self.user_hearing)
        workflow.add_node("human_feedback", self._human_feedback)
        workflow.add_node("topic_clarification", self.topic_clarification)
        workflow.add_node("query_generator", self.query_generator)
        workflow.add_node("search_executor", self.search_executor)
        workflow.add_node("relevance_filtering", self.relevance_filter)
        # workflow.add_node("web_page_reader", self.web_page_reader)
        workflow.add_node("document_summarizer", self.document_summarizer)
        workflow.add_node("summary_generator", self.summary_generator)
        
        
        # エントリーポイントと終了ポイントの設定
        workflow.set_entry_point("user_hearing")
        workflow.set_finish_point("summary_generator")

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
        self, state: SurveyAgentState
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
        
        try:
            # 最後のメッセージを安全に取得
            last_message = state["messages"][-1]
            
            # メッセージが辞書形式でない場合の対応
            if isinstance(last_message, dict):
                content = last_message.get("content", "")
            elif hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = str(last_message)
                
            # ユーザーへの質問を表示
            human_feedback = interrupt(content)
            
            # フィードバックがNoneの場合はデフォルトメッセージを使用
            if human_feedback is None:
                human_feedback = "そのままの条件で検索し、調査してください。"
                
            return Command(
                goto="user_hearing",
                update={
                    "messages": [
                        {"role": "assistant", "content": "ご要望を承りました。調査を続行します。"},
                        {"role": "human", "content": human_feedback}
                    ]
                },
            )
            
        except Exception as e:
            # エラーが発生した場合のフォールバック処理
            return Command(
                goto="user_hearing",
                update={
                    "messages": [
                        {"role": "assistant", "content": f"エラーが発生しました: {str(e)}。調査を最初からやり直します。"}
                    ]
                },
            )


graph = SurveyAgent(
    settings.fast_llm,
    # ArxivSearcher(settings.fast_llm),
).graph

if __name__ == "__main__":
    # searcher = ArxivSearcher(settings.fast_llm)
    agent = SurveyAgent(settings.fast_llm, searcher)
    initial_state: SurveyAgentState = {
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

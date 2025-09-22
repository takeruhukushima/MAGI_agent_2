import os

# import cohere
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScienceAgentSettings(BaseModel):
    """ScienceMagiエージェントの設定"""

    # 追加情報取得のための最大リトライ回数
    max_evaluation_retry_count: int = 3


class SurveyAgentSettings(BaseModel):
    """SurveyMagiエージェントの設定"""

    # タスク分解時の最小タスク数
    min_decomposed_tasks: int = 3
    # タスク分解時の最大タスク数
    max_decomposed_tasks: int = 5


class PlanningAgentSettings(BaseModel):
    """PlanningMagiエージェントの設定"""

    # 検索失敗時の最大リトライ回数
    max_retries: int = 3
    # 1回の検索で取得する最大論文数
    max_search_results: int = 10
    # 詳細分析する最大論文数
    max_papers: int = 3
    # PDFからmarkdown変換時の並列処理数
    max_workers: int = 3

class ExecutionAgentSettings(BaseModel):
    """ExecutionMagiエージェントの設定"""

    # 検索失敗時の最大リトライ回数
    max_retries: int = 3
    # 1回の検索で取得する最大論文数
    max_search_results: int = 10
    # 詳細分析する最大論文数
    max_papers: int = 3
    # PDFからmarkdown変換時の並列処理数
    max_workers: int = 3

class AnalysisAgentSettings(BaseModel):
    """AnalysisMagiエージェントの設定"""

    # 検索失敗時の最大リトライ回数
    max_retries: int = 3
    # 1回の検索で取得する最大論文数
    max_search_results: int = 10
    # 詳細分析する最大論文数
    max_papers: int = 3
    # PDFからmarkdown変換時の並列処理数
    max_workers: int = 3

class ReportAgentSettings(BaseModel):
    """ReportMagiエージェントの設定"""

    # 検索失敗時の最大リトライ回数
    max_retries: int = 3
    # 1回の検索で取得する最大論文数
    max_search_results: int = 10
    # 詳細分析する最大論文数
    max_papers: int = 3
    # PDFからmarkdown変換時の並列処理数
    max_workers: int = 3




class LangGraphSettings(BaseModel):
    """LangGraph関連の設定"""

    # ノードの最大実行回数制限
    max_recursion_limit: int = 1000


class ModelSettings(BaseModel):
    """モデル関連の設定"""

    # 高速・軽量なタスク用のOpenAIモデル
    openai_fast_model: str = "gpt-4o-mini"
    # 複雑なタスク用のOpenAIモデル
    openai_smart_model: str = "gpt-4o"
    # レポート生成用のAnthropicモデル
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    # 埋め込み用のOpenAIモデル
    openai_embedding_model: str = "text-embedding-3-small"
    # リランキング用のCohereモデル
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    # google gemini model
    google_gemini_fast_model: str = "gemini-2.5-flash"  # Updated to the correct model name
    # 生成時の温度パラメータ
    temperature: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # 必須のAPIキー
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    # COHERE_API_KEY: str
    # JINA_API_KEY: str
    GOOGLE_API_KEY: str
    TAVILY_API_KEY: str

    # LangSmith関連の設定
    LANGSMITH_TRACING_V2: str = "false"  # トレーシング機能の有効/無効
    LANGSMITH_ENDPOINT: str = (
        "https://api.smith.langchain.com"  # LangSmithのエンドポイント
    )
    LANGSMITH_API_KEY: str = ""  # LangSmithのAPIキー
    LANGSMITH_PROJECT: str = "arxiv-researcher"  # プロジェクト名

    # デバッグモード設定
    debug: bool = True

    # エージェント毎の設定インスタンス
    science_agent: ScienceAgentSettings = ScienceAgentSettings()
    survey_agent: SurveyAgentSettings = SurveyAgentSettings()
    planning_agent: PlanningAgentSettings = PlanningAgentSettings()
    execution_agent: ExecutionAgentSettings = ExecutionAgentSettings()
    analysis_agent: AnalysisAgentSettings = AnalysisAgentSettings()
    report_agent: ReportAgentSettings = ReportAgentSettings()
    langgraph: LangGraphSettings = LangGraphSettings()
    model: ModelSettings = ModelSettings()

    def __init__(self, **values):
        # .envファイルが存在する場合のみ読み込み（主に開発環境用）
        env_path = ".env"
        if os.path.exists(env_path):
            import dotenv

            dotenv.load_dotenv(env_path, override=True)
        super().__init__(**values)

    @property
    def llm(self) -> ChatGoogleGenerativeAI :
        """複雑なタスク用のLLMインスタンスを返す"""
        return ChatGoogleGenerativeAI(
            model=self.model.google_gemini_fast_model,
            temperature=self.model.temperature,
        )

    @property
    def fast_llm(self) -> ChatGoogleGenerativeAI:
        """高速・軽量なタスク用のLLMインスタンスを返す"""
        return ChatGoogleGenerativeAI(
            model=self.model.google_gemini_fast_model,
            temperature=self.model.temperature,
        )

    @property
    def reporter_llm(self) -> ChatGoogleGenerativeAI:
        """レポート生成用のLLMインスタンスを返す"""
        return ChatGoogleGenerativeAI(
            model=self.model.google_gemini_fast_model,
            temperature=self.model.temperature,
            max_tokens=8_192,
        )

    # @property
    # def cohere_client(self) -> cohere.Client:
    #     """Cohereクライアントインスタンスを返す"""
    #     return cohere.Client(api_key=self.COHERE_API_KEY)


# グローバルな設定インスタンス
settings = Settings()

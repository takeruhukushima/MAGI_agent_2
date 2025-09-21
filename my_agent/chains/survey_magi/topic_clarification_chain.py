from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.prompts import PromptManager


class TopicClarificationOutput(BaseModel):
    needs_clarification: bool = Field(
        default=False, 
        description="Whether the topic needs clarification"
    )
    question: str = Field(
        default="", 
        description="Clarification question to ask the user"
    )
    clarified_topic: str = Field(
        default="", 
        description="The clarified version of the topic if no clarification is needed"
    )


class TopicClarificationChain:
    """
    A chain that helps clarify and refine research topics.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize with an LLM instance."""
        self.llm = llm
        # Get the prompt template string from PromptManager
        prompt_template = PromptManager.get_prompt('survey', 'TOPIC_CLARIFIER')
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm.with_structured_output(
            TopicClarificationOutput,
            method="function_calling"
        )

    def __call__(self, state: dict) -> Command[Literal["topic_clarification", "query_generator"]]:
        """Execute the chain and determine next node."""
        print("--- [Chain] Survey MAGI: 1. Clarifying Topic ---")
        
        # 状態からトピックを取得（複数の可能性のあるキーをチェック）
        topic = state.get("research_theme") or state.get("topic")
        
        # メッセージからトピックを取得（もしあれば）
        if not topic and "messages" in state and state["messages"]:
            # 最後のユーザーメッセージを取得
            user_messages = [
                msg for msg in state["messages"] 
                if (hasattr(msg, 'type') and msg.type == 'human') or 
                   (isinstance(msg, dict) and msg.get('role') == 'user')
            ]
            
            if user_messages:
                last_user_message = user_messages[-1]
                if hasattr(last_user_message, 'content'):
                    topic = last_user_message.content
                elif isinstance(last_user_message, dict) and 'content' in last_user_message:
                    topic = last_user_message['content']
        
        # トピックが空またはデフォルトメッセージの場合は明確化が必要
        if not topic or topic == "調査テーマが指定されていません。詳細を教えてください。":
            return Command(
                goto="topic_clarification",
                update={
                    "messages": [{"role": "assistant", "content": "調査したい研究テーマについて教えていただけますか？"}],
                    "research_theme": None
                }
            )
        
        # トピックの明確化を実行
        clarification = self.run(topic, state.get("messages", []))
        
        if clarification.needs_clarification:
            return Command(
                goto="topic_clarification",
                update={
                    "research_theme": topic,  # 現在のトピックを保持
                    "clarification_question": clarification.question,
                    "messages": [{"role": "assistant", "content": clarification.question}]
                }
            )
            
        # 明確化されたトピックを取得
        final_topic = clarification.clarified_topic or topic
        
        # 最終的なトピックが有効なことを確認
        if not final_topic or final_topic == "調査テーマが指定されていません。詳細を教えてください。":
            return Command(
                goto="topic_clarification",
                update={
                    "messages": [{"role": "assistant", "content": "申し訳ありません、研究テーマを理解できませんでした。具体的にどのようなテーマについて調査しますか？"}],
                    "research_theme": None
                }
            )
            
        return Command(
            goto="query_generator", 
            update={
                "research_theme": final_topic,
                "topic": final_topic,  # 互換性のため両方のキーに設定
                "is_topic_clear": True,
                "messages": [
                    {"role": "assistant", "content": f"調査テーマを「{final_topic}」として進めます。"}
                ]
            }
        )

    def run(self, topic: str, messages: list[BaseMessage] = None) -> TopicClarificationOutput:
        """Run the topic clarification logic."""
        try:
            conversation_history = self._format_messages(messages) if messages else ""
            return self.chain.invoke({
                "research_theme": topic,  # 変数名をresearch_themeに変更
                "conversation_history": conversation_history
            })
        except Exception as e:
            raise RuntimeError(f"Topic clarification failed: {str(e)}")
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for the prompt."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# Create a single instance of the chain
from my_agent.settings import settings

topic_clarification_chain = TopicClarificationChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

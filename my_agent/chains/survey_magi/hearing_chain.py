from datetime import datetime
from typing import Literal, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.settings import settings
from my_agent.prompts import PromptManager


class Hearing(BaseModel):
    is_need_human_feedback: bool = Field(
        default=False, description="追加の質問が必要かどうか"
    )
    additional_question: str = Field(default="", description="追加の質問")


class HearingChain:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.prompt_manager = PromptManager()

    def __call__(self, state: dict) -> Command[Literal["human_feedback", "topic_clarification"]]:
        messages = state.get("messages", [])
        hearing = self.run(messages)
        message = []

        if hearing.is_need_human_feedback:
            message = [{"role": "assistant", "content": hearing.additional_question}]

        # 次に進むべきノードをここで決定する
        next_node = (
            "human_feedback" if hearing.is_need_human_feedback else "topic_clarification"
        )

        # 行き先(goto)とStateの更新内容(update)を持つCommandオブジェクトを返す
        return Command(
            goto=next_node,
            update={"hearing": hearing, "messages": message},
        )
    def run(self, messages: list[BaseMessage]) -> Hearing:
        try:
            prompt_template = self.prompt_manager.get_prompt('survey', 'HEARING_PROMPT')
            if not prompt_template:
                raise ValueError("Prompt 'HEARING_PROMPT' not found in survey prompts")
                
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm.with_structured_output(
                Hearing,
                method="function_calling",
            )
            hearing = chain.invoke(
                {
                    "current_date": self.current_date,
                    "conversation_history": self._format_history(messages),
                }
            )

            # --- ここにデバッグ用のprint文を追加！ ---
            print("--- DEBUG: HearingChain Output ---")
            print(f"is_need_human_feedback: {hearing.is_need_human_feedback}")
            print(f"additional_question: {hearing.additional_question}")
            print("---------------------------------")
            # ------------------------------------

        except Exception as e:
            raise RuntimeError(f"LLMの呼び出し中にエラーが発生しました: {str(e)}")

        return hearing
        
    def _format_history(self, messages: list[BaseMessage]) -> str:
        return "\n".join([f"{message.type}: {message.content}" for message in messages])


# Create a single instance of the chain
hearing_chain = HearingChain(
    llm=ChatGoogleGenerativeAI(
        model=settings.model.google_gemini_fast_model,
        temperature=settings.model.temperature
    )
)

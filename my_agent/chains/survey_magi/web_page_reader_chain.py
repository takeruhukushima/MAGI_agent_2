import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Literal
from langgraph.types import Command

class WebPageReaderChain:
    """
    URLリストを受け取り、各Webページの本文コンテンツを読み込むChain。
    """
    
    def __init__(self, llm=None):
        """
        初期化メソッド。
        
        Args:
            llm: 言語モデル（互換性のため保持）
        """
        self.llm = llm  # Store the llm for potential future use

    def __call__(self, state: dict) -> Command[Literal["document_summarizer"]]:
        """
        Stateからrelevant_documentsを取得し、各URLの内容を読み込んで更新します。
        """
        print("--- [Chain] Survey MAGI: 4a. Reading Web Page Content ---")

        documents = state.get("relevant_documents", [])
        if not documents:
            print("  > No documents to read.")
            # 読み込むドキュメントがない場合、更新せずに次のノードへ
            return Command(
                goto="document_summarizer",
                update={"relevant_documents": []}
            )

        # 各ドキュメントのコンテンツを更新
        updated_documents = self.run(documents)

        print(f"  > Successfully read content for {len(updated_documents)} documents.")

        # 全文で更新したドキュメントをstateに反映し、次のノードへ
        return Command(
            goto="document_summarizer",
            update={"relevant_documents": updated_documents}
        )

    def run(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        各ドキュメントのURLから本文を抽出し、'content'キーを更新します。
        """
        updated_docs = []
        for doc in documents:
            url = doc.get("url")
            title = doc.get("title", "No Title")
            print(f"  > Reading: {title[:70]}...")

            try:
                # Webページを取得して本文を抽出
                content = self._fetch_and_parse(url)

                if content:
                    # 元のドキュメント辞書をコピーし、contentを更新
                    updated_doc = doc.copy()
                    updated_doc["content"] = content
                    updated_docs.append(updated_doc)
                else:
                    print(f"    > Warning: Could not extract content from {url}")

            except Exception as e:
                print(f"    > Error reading URL {url}: {e}")
        
        return updated_docs

    def _fetch_and_parse(self, url: str) -> str:
        """
        指定されたURLからHTMLを取得し、主要なテキストコンテンツを抽出します。
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            # タイムアウトを設定してリクエスト
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # HTTPエラーがあれば例外を発生

            # BeautifulSoupでHTMLをパース
            soup = BeautifulSoup(response.content, 'html.parser')

            # scriptタグとstyleタグを削除
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            # テキストを取得し、行ごとに分割して余分な空白を削除
            lines = (line.strip() for line in soup.get_text().splitlines())
            # 空でない行を結合
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # 取得できるテキストが少なすぎる場合はスキップ
            if len(text) < 100:
                return ""
            
            return text

        except requests.exceptions.RequestException as e:
            print(f"      - Request failed: {e}")
            return ""
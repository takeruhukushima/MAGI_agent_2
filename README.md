# MAGI (Manus Artificial General Intelligence) プロジェクト

MAGIは、LangGraphを活用したマルチエージェント研究支援システムです。複数の専門エージェントが連携して、研究プロセス全体をサポートします。

## プロジェクト概要

MAGIは以下の4つの主要な専門エージェントで構成されています：

1. **Survey MAGI**: 研究テーマの明確化と文献調査を担当
2. **Planning MAGI**: 研究計画の立案と実験設計を担当
3. **Execution MAGI**: 実験の実行とデータ収集を担当
4. **Analysis MAGI**: データ分析と結果の解釈を担当

各エージェントは独立して動作する一方で、共通の状態（`AgentState`）を共有し、研究プロセスをシームレスに進行させます。

## ディレクトリ構造

```
my_agent/
├── chains/                  # 各エージェントのチェーン定義
│   ├── survey_magi/         # Survey MAGIのチェーン
│   ├── planning_magi/       # Planning MAGIのチェーン
│   ├── execution_magi/      # Execution MAGIのチェーン
│   └── analysis_magi/       # Analysis MAGIのチェーン
├── sub_agent/               # 各エージェントの実装
├── utils/                   # ユーティリティ関数
│   ├── nodes.py             # ノード定義
│   ├── state.py             # 状態管理
│   └── tools.py             # ツール定義
├── prompts.py               # プロンプトテンプレート
└── Science_MAGI.py          # メインのMAGIエージェント
```

## セットアップ

1. リポジトリをクローンします：
   ```bash
   git clone https://github.com/your-username/MAGI_agent_1.git
   cd MAGI_agent_1
   ```

2. 必要なパッケージをインストールします：
   ```bash
   pip install langgraph cli
   ```

3. 環境変数を設定します（.envファイルを作成）：
   ```
   TAVILY_API_KEY=your_tavily_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
4. LangGraphサーバーを起動します：
   ```bash
   langgraph up
   ```
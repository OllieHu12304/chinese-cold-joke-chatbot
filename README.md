# 繁體中文冷笑話與諧音梗助手

使用 Qwen3-1.7B 的 MLX 4-bit 版本，在 Apple M4 Pro 上以 QLoRA 微調多輪對話。
本專案實作語料設計、分組切分、參數高效微調與生成品質評估，並保留可重現的設定及原始輸出。

## 模型與工具來源

- [原始模型 Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)。
- [使用的 MLX 量化版本](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit)。模型卡標示 Apache-2.0。
- [MLX-LM 訓練工具](https://github.com/ml-explore/mlx-lm)。

選型階段比較 Qwen3-1.7B 與 Qwen2.5-3B-Instruct。兩者的初步笑話表現皆不足；
本實驗基於本機運算資源及反覆實驗成本，採用較小的 Qwen3。
選型原始觀察見 [model_comparison.md](model_comparison.md)。

## 本機環境

- Apple M4 Pro，24 GB 統一記憶體；macOS，arm64。
- Python 3.13.9。
- mlx-lm 0.32.2、mlx 0.32.4、transformers 5.20.2。
- 其餘相依版本由 `uv.lock` 固定；不需要安裝 CUDA 或 PyTorch。
- 模型 revision：`3b1b1768f8f8cf8351c712464f906e86c2b8269e`。

## 安裝與下載

在專案根目錄執行：

```bash
uv sync --locked
uv run python 00_download_model.py
```

`uv` 若未安裝，請先依 [uv 官方安裝文件](https://docs.astral.sh/uv/getting-started/installation/) 安裝。
模型下載到 `models/Qwen3-1.7B-4bit`，不納入 Git。
本 repository 的最佳 adapter 必須搭配上述基礎模型使用，不能單獨當成完整模型載入。

## 直接聊天

已完成訓練的本機版本可執行：

```bash
uv run python 07_chat.py
```

- `/reset` 清除對話歷史；`/exit` 結束。
- `uv run python 07_chat.py --base` 比較未微調模型。
- 程式會保存本機聊天原文至 `results/chat_sessions.jsonl`；公開發布前應排除私人對話。
- `messages` 傳遞前文，模型才有多輪上下文；這不是永久個人記憶。

## 語料

語料採用 AI 合成對話，未使用外部爬取或訪談資料。
資料包含三份基礎範例，以及 `data/joke_seeds.json` 中的 47 筆對話種子。
每筆含笑話、解釋、追問及回覆，包含中文近音梗、語意雙關、英文雙關及一般冷笑話。
AI 合成教材可能含生硬笑點與不自然表達，品質仍需要人工審查。

| 分組 | 完整對話數 | 前綴樣本數 |
|---|---:|---:|
| train | 40 | 118 |
| valid | 5 | 15 |
| test | 5 | 15 |

先按完整笑話分組，才展開各助手回答的前綴；同一來源不跨組。
三份開發階段已使用的基礎範例固定歸訓練集，不作為未見資料的評估依據。最長訓練樣本 175 tokens，不會觸發 512 token 上限。
分組清單、長度與檔案 SHA-256 見 `results/data_summary.json`。

## 微調方法

凍結 4-bit 基礎權重，在最後 16 個 Transformer 區塊的 q_proj、v_proj 加入 LoRA。
只更新約 0.918M 參數，占模型參數約 0.053%。這是量化基礎模型上的 LoRA，稱為 QLoRA。

| 設定 | 值 |
|---|---:|
| batch_size | 2 |
| iters | 200 |
| learning_rate | 0.0002 |
| LoRA rank / scale / dropout | 8 / 16 / 0.05 |
| seed | 42 |
| max_seq_length | 512 |
| 每幾步驗證 | 40 |

只計算最後助手回答的交叉熵，遮罩 prompt 與 padding。
`project_utils.py` 明確指定非思考模式；驗證 prompt 是完整 token 序列的前綴。
本次額外使用半開區間遮罩，避免 MLX-LM 0.32.2 預設 loss 將第一個 padding 位置計入。
這個邊界有單元測試，訓練與評估使用同一份 `answer_loss`。

每 40 步以全部 15 筆 valid 樣本評估並選最佳 adapter；test 不參與 checkpoint 選擇。
使用驗證 loss 選到的權重不一定是自由聊天品質最好的權重。

## 重現流程

```bash
uv run python 04_build_corpus.py
uv run python -m unittest discover -s tests -v
uv run python 05_train.py --smoke
uv run python 05_train.py
uv run python 06_evaluate.py --base
uv run python 06_evaluate.py
uv run python 08_report.py
```

`--smoke` 只跑 5 步且獨立輸出，不作為正式成果。
為避免覆蓋成果，訓練程式在已有權重時會停止；重跑前先將
`output/best_adapter`、`output/checkpoints` 或 `output/smoke` 改名備份。
重跑評估會覆寫 `evaluation_base.json`、`evaluation_adapted.json` 與報告，請先備份需要保留的實驗。

## 評估方式與限制

- base/adapter 使用相同 system、問題、temperature=0.7、top_p=0.8、top_k=20、seed=42、max_tokens=192。
- 原有開發四題與訓練重疊，僅供觀察，不宣稱是未見資料的測試。
- 獨立測試 loss 使用教材前文；解釋題亦以教材前文為條件，屬 teacher forcing。
- 自由三輪展示真正使用模型自己的上一輪回答，另外檢查訓練主題、留出主題與新主題。
- 小型合成測試集、單一種子及缺乏獨立人類盲評，限制了結果對廣泛中文或幽默能力的代表性。
- Loss 與 perplexity 下降不等於笑話更正確、更好笑；自由回答可能仍有重複、錯誤諧音或不合理解釋。
- 模型、硬體與套件版本改變，固定 seed 也不保證逐字重現。

## 結果檔案

本次實際訓練 200 步，約 114.5 秒，MLX 回報尖峰記憶體約 1.92 GB，
最佳權重為第 80 步。第 80 步後驗證 loss 回升，顯示過度擬合跡象。

| 指標 | 原始模型 | 最佳 adapter |
|---|---:|---:|
| valid loss | 5.9987 | 2.4282 |
| test loss | 6.6410 | 3.1022 |
| test perplexity | 765.85 | 22.25 |

**數值改善，但自由回答仍不可靠。**生成的笑話仍有語意不通、錯誤諧音與硬湊解釋，
訓練主題也未穩定掌握，因此尚不能支持笑話生成能力已改善的結論。
以下報告保留完整三輪輸出與失敗案例，模型回答未經改寫。

- `results/training_summary.json`：環境、來源、參數、時間、記憶體、最佳 step。
- `results/training_metrics.json`：訓練與驗證 loss 歷程。
- `results/evaluation_base.json`、`evaluation_adapted.json`：實際測試輸入與原始輸出。
- [完整報告與至少兩輪的實際對話](results/report.md)。
- [回答品質分析與評估限制](results/quality_review.md)。
- `output/best_adapter/`：選出的 LoRA 權重與設定，需搭配基礎模型。

## 程式閱讀順序

| 檔案 | 用途 |
|---|---|
| 01_try_model.py | 提供線索的單題推論實驗與紀錄 |
| 02_inspect_data.py | 單筆教材、模板與 token 數 |
| 03_prepare_data.py | 三份基礎教材的角色檢查與前綴切分 |
| 04_build_corpus.py | 正式資料驗證、分組與 token 檢查 |
| 05_train.py | QLoRA、loss 紀錄、最佳權重 |
| 06_evaluate.py | 公平基準比較與自由多輪展示 |
| 07_chat.py | 本機互動聊天 |
| 08_report.py | 將實際數值與回答產生為 Markdown |
| project_utils.py | 共用路徑、資料驗證、token 遮罩與模型載入 |

流程與原理說明見 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)，架構與驗證設計見 [docs/design.md](docs/design.md)。
原始 `results/experiments.jsonl` 未覆寫；其中一行缺少結尾大括號，修復副本為
`experiments_normalized.jsonl`，修復位置與原檔雜湊見 `history_repair.json`。

## 版本控制與發布內容

專案內容包含程式、語料、文件、評估結果與最佳 adapter。
虛擬環境、基礎模型及中途 checkpoint 由 `.gitignore` 排除；基礎模型可依固定 revision 重新下載。

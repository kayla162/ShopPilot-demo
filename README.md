# ShopPilot

[![tests](https://github.com/kayla162/ShopPilot-demo/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kayla162/ShopPilot-demo/actions/workflows/tests.yml)

把蝦皮賣家中心的報表變成商品層級的競品分析與優化建議；數字全部由程式計算，本機 AI 模型負責把結果寫成白話。

> Turns Shopee seller-center reports into product-level competitor analysis and rule-based recommendations. Numbers are computed deterministically (pandas, SQL, statistical tests); a local Qwen 2.5 1.5B model answers seller questions through **tool calling**, behind a numeric-grounding guard. A **QLoRA** fine-tune on program-generated trajectories (6 GB laptop GPU, 44 min) lifts tool-calling accuracy from 59% to **95%** on 141 auto-graded questions across three shops; **llama.cpp 4-bit** inference cuts CPU latency from 50 s to 3.6 s per answer; answers stream token by token, and every call is logged to a monitoring tab (p50/p95 latency, cache hits, guard blocks). 639 automated tests in this public repo.

- **線上 demo：** https://shoppilot-demo-fqegbgf2rvzuah3tzjhqqt.streamlit.app/
- **本機執行：** 見[快速開始](#快速開始)

## 解決什麼問題

蝦皮賣家中心可以下載商品表現、價格庫存、商品資料等報表，但它們是分開的 Excel 檔，賣家要自己對照才看得出：

- **哪些商品該優先處理？** 曝光高卻賣不動、缺貨中還有人在看、價格比同類貴很多……
- **跟誰比？** 自己店裡的同類商品、標題相似的商品，還是蝦皮上其他賣場的競品？
- **為什麼？要怎麼改？** 一個數字本身看不出原因，也不會告訴你下一步。

ShopPilot 把這些報表匯入同一個資料庫，替每個商品算出和三種對象（同類商品、相似商品、外部競品）的比較，產生**附上依據的規則式建議**，再讓 AI 把結果寫成白話，也可以直接用中文問「哪個類別賣最好？」「這件為什麼要補貨？」。

## 簡易動畫

在本機即時問 AI（Qwen 2.5 1.5B 本機推論；等待的部分以 2 倍速播放）：

![本機即時問答](docs/images/live-ai.gif)

| 總覽 | 競品分析 | AI 問答 |
|---|---|---|
| ![總覽](docs/images/overview.png) | ![競品分析](docs/images/competitor.png) | ![AI 問答](docs/images/qa.png) |

## 用了哪些技術？以及為什麼？

| 技術 | 用在哪裡 | 為什麼選它 |
|---|---|---|
| **Python、pandas** | 報表清理、彙整、每個商品的指標 | 蝦皮報表是 Excel，pandas 讀寫、分組計算最直接；分析核心是純 Python，可以單獨測試 |
| **SQLite** | 每家店一個資料庫 | 不用架伺服器、一個檔案就能帶著走；SQL 適合跨報表、跨期間的查詢 |
| **SciPy、NumPy** | z 檢定、Poisson、Mann-Whitney、Kruskal-Wallis、Spearman、OLS，Benjamini-Hochberg 校正 | 判斷「這個差距是不是真的」，樣本不足時標成「觀察中」，不憑感覺下結論 |
| **Streamlit、Plotly** | Dashboard 與互動圖表 | 用 Python 就能做互動畫面，並有 `AppTest` 可以自動測試畫面 |
| **Hugging Face `transformers` + Qwen 2.5 1.5B** | AI 文字解釋與問答 | 中文能力好、CPU 跑得動；**本機推論不用 API Key，賣場資料不離開電腦** |
| **Tool calling（函式呼叫）** | AI 問答：模型自己決定要查哪個商品、哪個排行 | 模型只看到它要的那一小段資料，不必在長表格裡找；查了什麼也能顯示給賣家看 |
| **PEFT LoRA** | 微調 1.5B 模型的工具用法 | 只訓練不到 2% 的參數，6 GB 筆電顯示卡就能訓練；訓練資料由程式產生，不用人工標註 |
| **llama.cpp（GGUF 量化）** | CPU 上的快速推論 | 4-bit 量化讓 CPU 回答從數十秒降到幾秒，權重記憶體少一半以上 |
| **OpenAI（選用）** | 同一個 provider 介面的另一種實作 | 需要更快、更穩定的回答時，改設定檔一行就能切換 |
| **sentence-transformers** | 外部競品自動配對的 embedding 評估 | 和不用模型的字元相似度用同一套指標比較，用數據決定要不要上模型 |
| **pytest、GitHub Actions** | 639 個自動化測試，每次 push 自動跑 | 分析結果是給賣家做決策用的，每個數字都要能被驗證 |
| **Docker、Streamlit Community Cloud、Git LFS** | 完整版容器、線上 demo、示範資料 | Docker 版內含模型、離線可用；線上 demo 免安裝就能看；資料庫與截圖用 LFS 存放 |

## 架構

```
蝦皮報表 (xlsx) ──▶ ingest ──▶ transform ──▶ SQLite ──▶ analytics ──▶ 結構化結果 (dict / JSON)
 檔名辨識、hash 去重   數值清理、標題規則   每家店一個 DB   統計、競品、規則          │
                                                                      ┌──────────────┴──────────────┐
                                                                  dashboard/                   llm/
                                                                  Streamlit 畫面               只讀結構化結果、寫成白話
```

| 資料夾 | 負責 |
|---|---|
| `shoppilot/ingest/` | 依檔名辨識報表類型和期間、讀取 Excel |
| `shoppilot/transform/` | 數值清理、標題拆解（標記、行銷語、核心標題） |
| `shoppilot/analytics/` | 商品指標、統計檢定、曝光因子、同類與相似商品、外部競品、規則式建議 |
| `shoppilot/llm/` | AI provider（本機 transformers / llama.cpp 量化 / OpenAI / 只讀快取）、prompt、數字檢查、問答（資料表或工具呼叫）、自動評估、LoRA 訓練資料、呼叫紀錄 |
| `shoppilot/match/` | 外部競品自動配對與評估 |
| `shoppilot/demo/` | 產生示範賣場資料、預先算好 AI 答案 |
| `dashboard/` | Streamlit 畫面 |

- `shoppilot/` 是純 Python 的分析核心，不 import Streamlit，也不 import 任何 AI 套件；拿掉整個 `llm/`，其他功能照常運作。
- 所有數字都由 Python、SQL、pandas 和統計檢定計算，**不需要 LLM 也能完整使用**。

## 工程亮點

### AI：數字由程式算，AI 只負責說清楚

- **本機推論：** `AutoModelForCausalLM` 加上 tokenizer 的 chat template 跑 `Qwen/Qwen2.5-1.5B-Instruct`；按下按鈕才載入，載入後留在記憶體重複使用，支援完全離線。
- **Greedy decoding（temperature 0）：** 相同輸入一定得到相同結果，所以可以快取，也最不容易自己編數字。
- **數字防捏造檢查：** AI 回覆裡的每個數字都要在輸入資料（或工具回傳的結果）裡找得到才會顯示。比對時考慮四捨五入、比例 ×100、百分比單位（把「多 N 件」寫成「高出 N%」會被擋下），並略過條列編號。
- **Provider 介面：** 本機 transformers、llama.cpp 量化模型、OpenAI、只讀快取實作同一個介面（含多輪對話、工具呼叫、串流），改一行設定或一個環境變數就能切換；各種錯誤都轉成固定訊息，不外洩例外內容。
- **線上 demo 的只讀快取 provider：** 線上 demo 不載入模型，而是用一個回報相同模型名稱的 provider，快取鍵（輸入資料＋模型＋prompt 版本）和本機預先算好的答案完全對得上，所以線上看到的就是同一個模型的真實輸出，點了馬上出現。

### AI 問答的兩種做法：資料表 vs. 工具呼叫

- **資料表（table）：** 依問題挑出全店摘要、類別排行、銷量排行、問題提到的商品，組成約 5 千字的資料表一次送給模型。欄位中文化、清單先排序、答案直接寫在清單旁邊（例如「銷量最高的類別」），實測從 5 題答對 1 題提升到 5 題。
- **工具呼叫（tools）：** 模型拿到 7 個工具（全店摘要、類別排行、銷量排行、查商品、搜尋商品、側欄選的商品、最該先處理的事），自己決定要查什麼；程式執行查詢、把結果交回去，最多 4 輪。模型只看到它要的那一小段資料，畫面上也會顯示「AI 查了：商品資料（A1234）」，賣家知道答案從哪裡來。

![工具呼叫：畫面上顯示 AI 查了哪些資料](docs/images/tools-qa.png)

- **工具的回傳也要為小模型設計：** 和資料表一樣把結論放在最前面（例如搜尋商品先寫「有，賣場有 22 個符合「洋裝」的商品」），長清單只留商品和問題；這些改法都只在開發集上比較、決定，不看評估題。
- **工具呼叫的穩健處理：** 解析 `<tool_call>` 時容忍多寫或少寫的括號、模型漏寫標籤；同樣的查詢重複出現或超過 4 輪就要求直接回答；不存在的工具和多餘的參數不會讓程式出錯。

### LoRA 微調：讓 1.5B 模型學會用工具

- **訓練資料由程式產生，不用人工標註：** 用和評估相同的方法，從示範資料算出「問題 → 正確的工具呼叫 → 工具回傳 → 標準回答」的完整對話（`shoppilot/llm/finetune.py`）：35 個訓練賣場、662 筆、13 種題型，每種題型有好幾種說法；標準回答只用工具回傳的數字，每一筆都先通過數字檢查才收。
- **不對著考題訓練：** 訓練賣場的亂數種子（101–130）和開發集（1–3）、評估的示範賣場（42）、真實賣場都不重疊；評估和開發集的原句不收進訓練資料（有測試檢查）。
- **6 GB 筆電顯示卡就能訓練（QLoRA）：** 原本的權重用 4-bit NF4 存、LoRA（rank 16，q/k/v/o/gate/up/down）用 bfloat16 訓練，只訓練 1,846 萬個參數（約 1.2%）；只算回答的 loss，只對要算 loss 的位置算詞彙表的 logits，gradient checkpointing。RTX 3060 Laptop（6 GB）1 個 epoch 43.7 分鐘，顯示卡記憶體最高 2.1 GB；驗證 loss 0.52 → 0.004。
- **用的時候合併進模型：** `config/llm.yaml` 的 `local.adapter` 指定權重資料夾，載入時用 PEFT 合併（`merge_and_unload`），推論速度和原本的模型一樣；快取鍵和監控紀錄的模型名稱會加上 `+lora:…`，不會和沒微調的模型混在一起。

**結果（同樣 141 題）：** 原本的模型會呼叫工具，但常選錯工具；微調後選對了，答對率從 59% 提升到 **95%**，超過資料表做法的 89%。改善最多的題型（示範賣場＋服飾店）：

| 題型 | 微調前 | 微調後 |
|---|---|---|
| 問側欄選的商品（「這件本期賣了幾件？」） | 0/12 | **12/12** |
| 商品銷量 | 4/10 | **10/10** |
| 長尾商品的銷量 | 0/10 | **10/10** |
| 商品類別 | 5/10 | **10/10** |
| 最該先處理的事 | 0/2 | **2/2** |

訓練用的全是程式產生的示範賣場（服飾），兩家真實賣場（服飾、保養品）都沒看過，照樣是 98% 和 89%。微調前的數字是 Q4 量化模型（CPU）、微調後是 float16（GPU）；量化在示範賣場只差 3 題（見下方「量化推論」的比較），和微調帶來的差距相比很小。

### 量化推論與串流輸出

- **llama.cpp（GGUF）：** 用 Qwen 官方的 4-bit 量化模型（Q4_K_M，1.1 GB）在 CPU 上推論。prompt 用同一個 Hugging Face tokenizer 的 chat template 產生（包含工具說明），和 transformers 版逐字相同，比較時只差在權重的精度。
- **串流輸出：** transformers 用自訂的 `TextStreamer`、llama.cpp 用 `stream=True`，一邊產生一邊顯示；工具呼叫那幾輪的 `<tool_call>` 文字會被過濾掉，只串流最後的回答。數字檢查要等整段寫完才做，沒通過時畫面會換成擋下的訊息。

同一個 Qwen 2.5 1.5B、同樣的資料表做法，在示範賣場 47 題（原本 32 題＋長尾 15 題）的比較（16 執行緒 CPU）：

| 推論方式 | 權重大小 | 原本的題目 | 長尾商品題 | 平均每題 |
|---|---|---|---|---|
| transformers float32 | 6.2 GB | 32/32 | 13/15 | 50.0 秒 |
| llama.cpp Q4_K_M | **1.1 GB** | 29/32 | 13/15 | **3.6 秒** |

4-bit 量化讓 CPU 上的回答快約 14 倍、權重小 5.6 倍；需要最高正確率時用 transformers（有顯示卡更快），需要在一般筆電上即時回答時用量化模型，改 `config/llm.yaml` 的 `provider` 就能切換。

### 可觀測性：AI 監控分頁

每次「AI 分析」「AI 問答」都寫一筆紀錄（延遲、輸入／輸出 tokens、快取命中、被數字檢查擋下、錯誤原因、工具呼叫次數、模型呼叫輪數），「AI 監控」分頁顯示快取命中率、延遲 p50／p95、擋下比例、錯誤率、輸出速度，並依模型、依問答做法分組。延遲只算真的呼叫了模型的紀錄，快取命中不算進去，免得數字看起來太好。寫紀錄失敗不影響回答。

### 評估：題目和答案都由程式算出

- **自動出題、自動評分：** 題目和標準答案都由程式從同一份資料算出（賣最好的類別、某商品的價格／類別／銷量、某類別賣最好的商品、側欄選的商品、賣場沒有的東西…），換賣場、換期間都能重跑。
- **長尾商品題：** 原本的商品題只問銷量前 5 名；另外加上銷量 31 名以後的商品、用簡稱問價格，分開計分（在看任何結果之前就定好）。
- **開發集與評估分開：** 改 prompt、資料表、工具回傳時只看開發集（換亂數種子的示範賣場、換句話說的問法）；LoRA 的訓練資料用另外的亂數種子，也不收評估的原句。
- **示範賣場任何人都能重跑：** `python -m shoppilot.llm.qa_eval --demo --extended --mode tools`；另外兩家是真實賣場，資料不公開。

3 個賣場（示範賣場＋兩家真實賣場）、每家 32 題原本的題目＋15 題長尾商品題，共 141 題：

| 做法 · 模型 | 原本的題目（96 題） | 長尾商品題（45 題） | 合計 |
|---|---|---|---|
| 資料表 · Qwen 2.5 1.5B（Q4，CPU） | 87（91%） | 38（84%） | 125（89%） |
| 工具呼叫 · Qwen 2.5 1.5B（Q4，CPU） | 55（57%） | 28（62%） | 83（59%） |
| **工具呼叫 · Qwen 2.5 1.5B ＋ LoRA（float16，GPU）** | **89（93%）** | **45（100%）** | **134（95%）** |

各賣場的「答對」都只檢查回答裡有沒有標準答案（商品、類別、數字），每一題的問題、標準答案、回答原文都存在 `details.csv`。

### 外部競品自動配對：用數據決定要不要上模型

比較四種計分（字元雙字詞 bigram、`multilingual-e5-small`、`multilingual-e5-base`、`bge-m3`）和兩種標題（原樣 / 去掉【現貨】、#hashtag 的核心標題），用賣家確認的同款配對算 Recall@k、MRR，並以 bootstrap 算 95% 信賴區間：

| 計分 | 標題 | Recall@1 | Recall@3（95% 區間） | Recall@5 | MRR（95% 區間） | 每筆（毫秒） |
|---|---|---|---|---|---|---|
| bigram | raw | 38% | 69%（50%–88%） | 75% | 0.54（0.36–0.73） | 2.9 |
| bigram | core | 31% | 62%（38%–88%） | 75% | 0.50（0.32–0.68） | 1.7 |
| e5-small | raw | 31% | 50%（25%–75%） | 56% | 0.44（0.26–0.63） | 31.9 |
| e5-small | core | 19% | 50%（25%–75%） | 69% | 0.41（0.25–0.58） | 16.3 |
| e5-base | raw | 19% | 38%（12%–62%） | 56% | 0.35（0.19–0.53） | 101.0 |
| e5-base | core | 25% | 69%（44%–88%） | 69% | 0.47（0.30–0.65） | 55.6 |
| bge-m3 | raw | 44% | 62%（38%–81%） | 69% | 0.57（0.39–0.77） | 1445.2 |
| bge-m3 | core | 44% | 62%（38%–88%） | 62% | 0.56（0.36–0.76） | 346.8 |

Recall@3 最高的是 bigram/raw 和 e5-base/core（都是 69%），依事先定好的規則選較小、較快的 **bigram**：不用模型、每筆約 3 毫秒，以「列出前 3 名建議、由賣家確認」的方式上線。

### 資料工程

- **冪等匯入：** 依檔名辨識報表類型和期間，用檔案 SHA-256 去重；範圍外的檔案（例如含個資的訂單檔）只看檔名、不打開；單一檔案失敗不影響其他檔案。
- **資料模型：** 期間型資料（商品表現）和快照型資料（價格庫存、商品資料）分開存，另有外部競品的歷次「已售出」記錄表和 AI 快取表；寫入由呼叫端控制交易。
- **多賣場隔離：** 每家店一個資料夾、一個資料庫，可以有自己的類別、屬性詞、門檻設定；賣場名稱做路徑安全檢查。
- **資料品質：** 已刪除商品的銷量另外記錄，並有測試檢查「分析的銷量＋已刪除商品的銷量＝報表總數」。
- **同期銷量推算：** 外部競品只有累計「已售出」，用每週記錄的差距推算它在報表期間的銷量，才能和自己的期間銷量比。
- **示範資料：** 用固定亂數種子產生和蝦皮相同格式的報表（150 個商品、10 個類別），埋入幾個規律（標記多→曝光低、缺貨多→銷量低、同類偏貴→轉換低），走和真實報表完全相同的匯入流程。

### 測試

- 這個公開版有 639 個自動化測試（`python -m pytest`）：單元測試、Dashboard 畫面測試（Streamlit `AppTest`），以及用示範資料跑完整流程的測試；AI 相關測試用假的模型，不必下載模型。
- 每次 push 都由 GitHub Actions 在 Ubuntu、Python 3.13 上只裝 `requirements.txt` 跑完整測試（見開頭徽章）；套件版本固定，和線上 demo 安裝的完全相同。
- 私人版另有一組**真實資料測試**：用每家店每個期間的真實報表檢查匯入、加總、每個商品的分析、建議依據，並實際切到該期間檢查 Dashboard 各分頁。

## 快速開始

### 線上 demo

直接打開 https://shoppilot-demo-fqegbgf2rvzuah3tzjhqqt.streamlit.app/ 。資料是虛構的示範服飾賣場；AI 問答分頁的範例問題和側欄預設商品的「AI 分析」，點了馬上出現。久沒人開時會先顯示喚醒按鈕，按下後約 1 分鐘啟動。

### 本機執行

示範資料庫、xlsx、截圖用 Git LFS 存放，**clone 之前先執行一次 `git lfs install`**（已經 clone 的話，裝好 git-lfs 後執行 `git lfs pull`）。

```bash
git lfs install
git clone https://github.com/kayla162/ShopPilot-demo.git
cd ShopPilot-demo
pip install -r requirements.txt
mkdir -p /tmp/shoppilot-demo
cp demo/shoppilot.db demo/examples.json /tmp/shoppilot-demo/    # 複製一份再用，不改到版控裡的 demo/
SHOPPILOT_DATA_DIR=/tmp/shoppilot-demo streamlit run dashboard/app.py
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$env:TEMP\shoppilot-demo" | Out-Null
Copy-Item demo\shoppilot.db, demo\examples.json "$env:TEMP\shoppilot-demo\"
$env:SHOPPILOT_DATA_DIR = "$env:TEMP\shoppilot-demo"
streamlit run dashboard/app.py
```

打開 http://localhost:8501。加上環境變數 `SHOPPILOT_DEMO=1` 就是和線上 demo 相同的展示模式。

**即時問 AI：** 再安裝本機模型需要的套件，第一次用到 AI 時會下載約 3 GB 的模型；有 NVIDIA 顯示卡的話裝 CUDA 版 torch 會快很多。

```bash
pip install -r requirements-local.txt
```

**用量化模型（CPU 上快很多）：** 第一次使用時下載約 1.1 GB 的 GGUF。

```bash
pip install -r requirements-local.txt -r requirements-quant.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

再把 `config/llm.yaml` 的 `provider` 改成 `llamacpp`。

**AI 問答改用工具呼叫：** `config/llm.yaml` 的 `qa_mode` 改成 `tools`。

**訓練 LoRA（需要 NVIDIA 顯示卡）：**

```bash
pip install -r requirements-train.txt   # torch 要裝 CUDA 版，見檔案開頭的說明
python scripts/train_lora.py --qlora --epochs 1      # 輸出到 models/shoppilot-tools-lora/
python -m shoppilot.llm.qa_eval --demo --extended --mode tools --provider local:Qwen/Qwen2.5-1.5B-Instruct@models/shoppilot-tools-lora
```

要在 Dashboard 使用，在 `config/llm.yaml` 的 `local` 加上 `adapter: models/shoppilot-tools-lora`，`qa_mode` 改成 `tools`。

想改用 OpenAI：`pip install -r requirements-ai.txt`，把 `config/llm.yaml` 的 `provider` 改成 `openai`，在專案根目錄的 `.env` 寫 `OPENAI_API_KEY=...`。

### Docker（含模型的完整版）

```bash
docker build -t shoppilot .
docker run -p 7860:7860 shoppilot
```

打開 http://localhost:7860。映像建置時就下載好模型，執行時不需要網路，啟動時在背景先載入模型，可以直接即時問 AI。

## 深入閱讀

| 想看什麼 | 從這裡開始 |
|---|---|
| 本機模型怎麼載入、推論 | `shoppilot/llm/local_provider.py` |
| 數字防捏造檢查、AI 文字解釋 | `shoppilot/llm/explainer.py` |
| AI 問答怎麼依問題組資料表 | `shoppilot/llm/qa.py`、`shoppilot/llm/prompts/shop_qa.md` |
| 自動出題、自動評分 | `shoppilot/llm/qa_eval.py`（`python -m shoppilot.llm.qa_eval`） |
| AI 問答的工具與工具呼叫迴圈 | `shoppilot/llm/qa_tools.py`、`shoppilot/llm/prompts/shop_qa_tools.md` |
| LoRA 訓練資料與訓練 | `shoppilot/llm/finetune.py`、`scripts/train_lora.py` |
| 量化推論（llama.cpp） | `shoppilot/llm/llamacpp_provider.py` |
| 呼叫紀錄與監控指標 | `shoppilot/llm/telemetry.py`、`dashboard/views/ai_monitor.py` |
| 線上 demo 的只讀快取 provider | `shoppilot/llm/cache_only_provider.py` |
| 模型比較 | `shoppilot/llm/benchmark.py`（`python -m shoppilot.llm.benchmark`） |
| 外部競品配對與評估 | `shoppilot/match/`（`python -m shoppilot.match.evaluate`） |
| 統計檢定與規則式建議 | `shoppilot/analytics/stats.py`、`shoppilot/analytics/rules.py` |
| 報表辨識與冪等匯入 | `shoppilot/ingest/`、`shoppilot/pipeline.py` |
| 示範資料怎麼產生 | `shoppilot/demo/generate.py` |
| 規則與門檻設定 | `config/keywords.yaml`、`config/rules.yaml`、`config/llm.yaml` |

## 授權

程式碼以 [MIT License](LICENSE) 授權。

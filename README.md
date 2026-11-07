# ShopPilot

[![tests](https://github.com/kayla162/ShopPilot-demo/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kayla162/ShopPilot-demo/actions/workflows/tests.yml)

把蝦皮賣家中心的報表變成商品層級的競品分析與優化建議；數字全部由程式計算，本機 AI 模型負責把結果寫成白話。

> Turns Shopee seller-center reports into product-level competitor analysis and rule-based recommendations. Numbers are computed deterministically (pandas, SQL, statistical tests); a local Hugging Face model (Qwen 2.5 1.5B) rewrites them in plain language behind a numeric-grounding guard. 582 automated tests in this public repo.

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
| **OpenAI（選用）** | 同一個 provider 介面的另一種實作 | 需要更快、更穩定的回答時，改設定檔一行就能切換 |
| **sentence-transformers** | 外部競品自動配對的 embedding 評估 | 和不用模型的字元相似度用同一套指標比較，用數據決定要不要上模型 |
| **pytest、GitHub Actions** | 582 個自動化測試，每次 push 自動跑 | 分析結果是給賣家做決策用的，每個數字都要能被驗證 |
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
| `shoppilot/llm/` | AI provider（本機 / OpenAI / 只讀快取）、prompt、數字檢查、問答與自動評估 |
| `shoppilot/match/` | 外部競品自動配對與評估 |
| `shoppilot/demo/` | 產生示範賣場資料、預先算好 AI 答案 |
| `dashboard/` | Streamlit 畫面 |

- `shoppilot/` 是純 Python 的分析核心，不 import Streamlit，也不 import 任何 AI 套件；拿掉整個 `llm/`，其他功能照常運作。
- 所有數字都由 Python、SQL、pandas 和統計檢定計算，**不需要 LLM 也能完整使用**。

## 工程亮點

### AI：數字由程式算，AI 只負責說清楚

- **本機推論：** `AutoModelForCausalLM` 加上 tokenizer 的 chat template 跑 `Qwen/Qwen2.5-1.5B-Instruct`；按下按鈕才載入（Docker 版啟動時在背景預先載入），載入後留在記憶體重複使用，支援完全離線。
- **Greedy decoding（temperature 0）：** 相同輸入一定得到相同結果，所以可以快取，也最不容易自己編數字。
- **數字防捏造檢查：** AI 回覆裡的每個數字都要在輸入資料裡找得到才會顯示。比對時考慮四捨五入、比例 ×100、百分比單位（把「多 N 件」寫成「高出 N%」會被擋下），並略過條列編號。
- **依問題組資料表：** 不把整個資料庫丟給模型，而是挑出問題提到的商品、類別，加上全店摘要、各類別排行、重點建議，組成約 5 千字的資料表。欄位中文化、清單先排序之後，實測從 5 題答對 1 題提升到 5 題。
- **自動評估：** 題目和標準答案都由程式從同一份資料算出，換賣場、換期間都能重跑、自動評分。本機 Qwen 1.5B 在兩家真實賣場：服飾店 **92%**（23/25）、保養品店 **88%**（22/25）。
- **用開發集調整、不碰評估題：** 另外產生換亂數種子的開發集，比較十多種資料表寫法再套用，避免對著評估題調參。例如「在銷量排行旁直接寫出第一名」讓開發集 28/37 → 35/37、評估 76 → 85 題答對；處理「銷量 0」的寫法讓開發集 10/30 → 26/30、評估 85 → 89。
- **Provider 介面：** 本機模型、OpenAI、只讀快取實作同一個介面，改一行設定或一個環境變數就能切換；各種錯誤都轉成固定訊息，不外洩例外內容。
- **線上 demo 的只讀快取 provider：** 線上 demo 不載入模型，而是用一個回報相同模型名稱的 provider，快取鍵（輸入資料＋模型＋prompt 版本）和本機預先算好的答案完全對得上，所以線上看到的就是同一個模型的真實輸出，點了馬上出現。
- **模型比較：** 同一批商品比較 Qwen 0.5B、1.5B、OpenAI 的數字捏造率、格式符合率、回應時間、成本和記憶體。

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

- 這個公開版有 582 個自動化測試（`python -m pytest`）：單元測試、Dashboard 畫面測試（Streamlit `AppTest`），以及用示範資料跑完整流程的測試；AI 相關測試用假的模型，不必下載模型。
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
| 線上 demo 的只讀快取 provider | `shoppilot/llm/cache_only_provider.py` |
| 模型比較 | `shoppilot/llm/benchmark.py`（`python -m shoppilot.llm.benchmark`） |
| 外部競品配對與評估 | `shoppilot/match/`（`python -m shoppilot.match.evaluate`） |
| 統計檢定與規則式建議 | `shoppilot/analytics/stats.py`、`shoppilot/analytics/rules.py` |
| 報表辨識與冪等匯入 | `shoppilot/ingest/`、`shoppilot/pipeline.py` |
| 示範資料怎麼產生 | `shoppilot/demo/generate.py` |
| 規則與門檻設定 | `config/keywords.yaml`、`config/rules.yaml`、`config/llm.yaml` |

## 授權

程式碼以 [MIT License](LICENSE) 授權。

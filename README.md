# ShopPilot

[![tests](https://github.com/kayla162/ShopPilot-demo/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kayla162/ShopPilot-demo/actions/workflows/tests.yml)

把蝦皮賣家中心的報表變成商品層級的競品分析與優化建議；數字全部由程式計算，本機 AI 模型只負責把結果寫成白話。

- **線上 demo（Streamlit Community Cloud）：** https://shoppilot-demo-fqegbgf2rvzuah3tzjhqqt.streamlit.app/
- **本機執行：** 見下方[本機執行](#本機執行)

![總覽](docs/images/overview.png)
![競品分析](docs/images/competitor.png)
![AI 問答](docs/images/qa.png)

在本機即時問 AI（Qwen 2.5 1.5B 本機推論；問題不在快取裡，等待的部分以 2 倍速播放）：

![本機即時問答](docs/images/live-ai.gif)

## English summary

ShopPilot turns Shopee seller-center reports into product-level competitor analysis and rule-based recommendations, served as a Streamlit dashboard. Every number is computed deterministically (pandas, SQL, statistical tests); an optional LLM layer only rewrites those results in plain language. The LLM runs locally by default via Hugging Face `transformers` (Qwen 2.5 1.5B), behind a provider interface that can switch to OpenAI. A numeric-grounding guard blocks any reply containing numbers absent from the input, and an auto-generated, auto-graded Q&A benchmark scores 92% / 88% on two real shops. Data side: idempotent hash-based ingestion, per-shop SQLite isolation, period-aware analysis, and 576 automated tests in this public repo. The free online demo (Streamlit Community Cloud) cannot run the model, so it serves answers precomputed by the same local model through a cache-only provider whose cache keys match; live inference runs locally or in Docker. All data in this repo and in the online demo is synthetic (a fictional clothing shop generated with a fixed random seed); the real shops' reports are not published.

## 技術重點

### 架構

```
蝦皮報表 (xlsx) ──▶ ingest ──▶ transform ──▶ SQLite ──▶ analytics ──▶ 結構化結果 (dict / JSON)
 檔名辨識、hash 去重   數值清理、標題規則   每家店一個 DB   統計、競品、規則          │
                                                                      ┌──────────────┴──────────────┐
                                                                  dashboard/                   llm/（選用）
                                                                  Streamlit 畫面               只讀結構化結果、寫成白話
```

- `shoppilot/` 是純 Python 的分析核心，不 import Streamlit，也不 import 任何 AI 套件；拿掉整個 `llm/` 資料夾，其他功能照常運作。
- **不需要 LLM 也能完整使用。** 所有數字都由 Python、SQL、pandas 和統計檢定計算。不做評論分析，也不爬蝦皮網站。

### AI：本機 Hugging Face 模型，以及怎麼讓它不亂說

設計原則是**數字全部由程式算，AI 只負責把結果寫成白話**，AI 出錯也不影響核心分析。

- **本機推論（`shoppilot/llm/local_provider.py`）：** 用 Hugging Face `transformers` 的 `AutoModelForCausalLM` 和 tokenizer 的 chat template 跑 `Qwen/Qwen2.5-1.5B-Instruct`。選它是因為中文能力好、CPU 跑得動。本機執行時按下按鈕才 import 和載入（Docker 版則在啟動時於背景預先載入），載入後留在記憶體重複使用；支援離線模式，展示時不需要網路。
- **Greedy decoding（temperature 0）：** 每一步取機率最高的字，相同輸入一定得到相同結果，可以快取，也最不容易自己編數字。線上 demo 的範例問題就是事先用同一個模型算好、存進快取的。
- **只讀快取的 provider（`shoppilot/llm/cache_only_provider.py`）：** 免費主機的記憶體跑不動模型，所以線上 demo 換成一個不跑模型的 provider。它回報和本機模型相同的名稱，快取鍵（輸入資料＋模型＋prompt 版本）就對得上預先算好的答案；快取裡沒有的問題回固定說明，不會假裝回答。用環境變數 `SHOPPILOT_LLM_PROVIDER=cache_only` 切換，不用改設定檔。
- **Provider 介面：** 本機模型和 OpenAI 實作同一個介面，改 `config/llm.yaml` 一行就能切換；錯誤（沒裝套件、模型載入失敗、記憶體不足、API 錯誤）都轉成固定訊息，不外洩例外內容。
- **數字防捏造檢查：** AI 回覆裡的每個數字都要在輸入資料裡找得到，否則整段不顯示。比對時考慮四捨五入、比例 ×100、百分比單位（把「多 N 件」寫成「高出 N%」會擋下，這是實測抓到的例子），並略過行首的條列編號。
- **AI 問答：** 不把整個資料庫丟給模型，而是依問題挑資料：問題提到的商品、類別，加上全店摘要、各類別（已依銷量排序）、重點建議，組成約 5 千字的資料表。例如在示範賣場問「哪個類別賣最好？」，資料表只帶各類別的銷量排行和全店摘要。把欄位改成中文、清單先排序之後，實測從 5 題答出 1 題提升到 5 題。
- **自動評估（`shoppilot/llm/qa_eval.py`）：** 題目和標準答案都由程式從同一份資料算出，換賣場、換期間都能重跑、自動評分。本機 Qwen 1.5B：服飾店 23/25（92%）、保養品店 22/25（88%）——這兩個數字是在兩家真實賣場測的，資料不公開；示範資料可以自己重跑。錯誤分析：錯的都是「把別的商品的數字套到這個商品」，數字都在資料裡，所以數字檢查擋不到，這點寫在 [docs/04_已知限制.md](docs/04_已知限制.md)。評估可以加 `--sidebar` 模擬 Dashboard（每題都帶側欄選的商品），這樣測出問全店的事時，小模型會答成側欄的商品；改成問到「這個商品」才帶側欄商品之後，3 份資料 72 題中「回答提到側欄商品、但它不是答案」從 6 題降到 1 題，不管側欄選什麼答對率都是 82%（修正前 81%–85%：側欄剛好是答案時比較高，所以預設的側欄商品少答對 2 題）。之後又補上「這個」「那個商品」「他」這些說法：在側欄選了商品後這樣問的 9 題從 2 題答對變成 7 題，用「它」指題目裡的商品時也不再答成側欄的商品；再把側欄的商品放在資料表自己的標題底下，「這件本期賣了幾件？」從 0/3 變成 2/3。細節見已知限制第 11 項。
- **模型比較（`shoppilot/llm/benchmark.py`）：** 同一批商品比較 Qwen 0.5B、1.5B、OpenAI 的數字捏造率、格式符合率、回應時間、成本和記憶體。

### 資料工程

- **冪等匯入：** 依檔名辨識報表類型和期間，用檔案 SHA-256 去重；範圍外的檔案（例如含個資的訂單檔）只看檔名、不打開；單一檔案失敗不影響其他檔案，失敗的檔案可以重試。
- **資料模型：** SQLite，期間型資料（商品表現）和快照型資料（價格庫存、商品資料）分開存，另有外部競品的歷次「已售出」記錄表和 AI 快取表；寫入由呼叫端控制交易。
- **多賣場隔離：** 每家店一個資料夾、一個資料庫，可以有自己的設定檔（類別、屬性詞、門檻）；賣場名稱做路徑安全檢查。
- **資料品質：** 報表裡已刪除的商品沒有名稱和價格，分析不使用，但會記下件數，並有測試檢查「分析的銷量＋已刪除商品的銷量＝報表總數」。示範資料也放了幾個已刪除的商品。
- **同期銷量推算：** 外部競品只有累計「已售出」，用每週記錄的差距推算報表期間的銷量；顯示成約數（例如「1.2萬」）、記錄日期對不上的不推算。示範資料的外部競品有期間頭尾兩次記錄，可以看到推算結果。
- **示範資料（`shoppilot/demo/`）：** 用固定亂數種子產生和蝦皮相同格式的報表（150 個一般商品、10 個類別），刻意埋入幾個規律（標記多→曝光低、缺貨多→銷量低、同類偏貴→轉換低），讓曝光因子和規則建議有結果可以展示；走和真實報表完全相同的匯入流程。

### 分析方法

- 統計工具：z 檢定、Poisson、Mann-Whitney、Kruskal-Wallis、Spearman、OLS，多重比較用 Benjamini-Hochberg 校正。
- 規則式建議都附上程式算好的依據；樣本不足時標成「觀察中」，不下結論。
- 三種比較對象：同類商品、標題相似商品（字元雙字詞相似度）、外部競品。

### 外部競品自動配對（embedding）

- **為什麼需要：** 外部競品要先對到自己的哪個商品才能比較。賣家確認是同款的競品，標題的字元相似度只有 0.02–0.14，很難用門檻判斷，所以改成「排名、列出前 3 名建議」，由賣家確認。
- **做法：** 四種計分——字元雙字詞相似度（bigram，不用模型）、`multilingual-e5-small`、`multilingual-e5-base`、`bge-m3`（sentence-transformers，本機執行）；兩種標題——原樣（raw）和去掉【現貨】、#hashtag 等標記的核心標題（core）。用賣家確認的同款配對算 Recall@k、MRR，並以 bootstrap 算 95% 信賴區間（`python -m shoppilot.match.evaluate`）。
- **真實賣場的比較結果**（16 組標註，略過 0 組）：

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

- **選定 bigram＋原樣標題：** Recall@3 最高的是 bigram/raw 和 e5-base/core（都是 69%），區間幾乎重疊，依事先定好的規則選較小、較快的——bigram 不用模型。bge-m3、e5-small 和 e5-base/core 的平均名次與最差名次比 bigram/raw 好（正解較少被排到很後面），e5-base/raw 則較差；整體在這份資料上沒有明顯勝出。
- **說明：** 標註只有 16 組，區間很寬，差距可能只是運氣。同分時取樂觀名次（名次＝分數嚴格較高的商品數＋1），bigram 很多商品同為 0 分，這種算法對它較有利。「每筆」是整批計算的平均（GPU），不是單筆延遲。真實賣場的資料不公開；示範資料可以自己重跑。

### 測試

- 這個公開版有 576 個自動化測試（`python -m pytest`）：單元測試、Dashboard 畫面測試（Streamlit `AppTest`），以及用示範資料跑完整流程的測試。
- 每次 push 都由 GitHub Actions（`.github/workflows/tests.yml`）在 Ubuntu、Python 3.13 上只裝 `requirements.txt` 跑完整測試，結果見開頭的徽章。`requirements.txt` 是固定版本，和線上 demo 安裝的完全相同。
- 私人版另外有一組**真實資料測試**：用每家店每個期間的真實報表（複製到暫存資料夾，不動真正的資料庫）檢查匯入、加總、每個商品的分析、建議依據、AI 資料表大小，並實際切到該期間檢查 Dashboard 各分頁。這些測試綁定真實資料，不放進公開版。
- AI 相關的測試用假的模型，不需要下載模型也能跑。

## 線上 demo 說明

- 放在 Streamlit Community Cloud（免費）：唯讀的展示模式，資料是虛構的示範賣場，不能上傳報表。
- 免費主機跑不動模型，所以 AI 只提供**預先算好的答案**：AI 問答分頁的範例問題，以及側欄預設商品等幾個商品的「AI 分析」，點了馬上出現；自己輸入問題要在本機或用 Docker 執行。
- 閒置一段時間沒人用會休眠，打開時按一下喚醒，等約 1 分鐘。

## 本機執行

示範資料庫（`demo/shoppilot.db`）、xlsx、截圖都用 Git LFS 存放，**clone 之前要先執行一次 `git lfs install`**；否則拿到的只是 LFS 指標檔，Dashboard 會無法開啟資料庫。（已經 clone 的話，裝好 git-lfs 後在專案裡執行 `git lfs pull`。）

```bash
git lfs install
git clone <這個 repo 的網址>
cd <clone 下來的資料夾>
pip install -r requirements.txt
mkdir -p /tmp/shoppilot-demo
cp demo/shoppilot.db demo/examples.json /tmp/shoppilot-demo/    # 複製一份再用，不要改到版控裡的 demo/
SHOPPILOT_DATA_DIR=/tmp/shoppilot-demo streamlit run dashboard/app.py
```

`demo/` 有進版控，Dashboard 執行時會寫入快取，所以先把 `demo/shoppilot.db` 和 `demo/examples.json` 複製到暫存資料夾，再把 `SHOPPILOT_DATA_DIR` 指到那份複本（和 `docker/start.sh` 的做法一樣）；示範資料已經建好，不需要重新產生。Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$env:TEMP\shoppilot-demo" | Out-Null
Copy-Item demo\shoppilot.db, demo\examples.json "$env:TEMP\shoppilot-demo\"
$env:SHOPPILOT_DATA_DIR = "$env:TEMP\shoppilot-demo"
streamlit run dashboard/app.py
```

頁面在 http://localhost:8501。再加上環境變數 `SHOPPILOT_DEMO=1`，就是和線上 demo 相同的唯讀展示模式（AI 問答分頁有預先算好的範例問題）。想用 AI 文字解釋和問答的話，再安裝本機模型需要的套件（第一次用到 AI 時會下載約 3 GB 的模型；用 CPU 回答一題約 30–60 秒）：

```bash
pip install -r requirements-local.txt
```

有 NVIDIA 顯示卡的話，裝 CUDA 版 torch 會快很多；也可以改用 OpenAI（`pip install -r requirements-ai.txt`，把 `config/llm.yaml` 的 `provider` 改成 `openai`，在專案根目錄的 `.env` 寫 `OPENAI_API_KEY=...`）。

## Docker

```bash
docker build -t shoppilot .
docker run -p 7860:7860 shoppilot
```

打開 http://localhost:7860。這是**可以即時問 AI 的完整版**：映像建置時就下載好模型（約 3 GB），執行時不連網、啟動時在背景先載入模型。

## 部署（維護者）

1. 先執行一次 `git lfs install`（示範資料庫、xlsx、截圖都用 Git LFS；Streamlit Community Cloud 支援 LFS）。
   這個 repo 由私人版的 `scripts/export_public.py` 產生，每次匯出都會在 `main` 加一個新 commit。**第一次推送時 `main` 應該只有一個 commit**：推送前如果已經匯出過好幾次，先刪掉這個資料夾（或它的 `.git`）再重新匯出一次。之後每次匯出的新 commit 照常推送即可。
2. GitHub：`git remote add origin <GitHub repo>`，再 `git push -u origin main`。
3. Streamlit Community Cloud（share.streamlit.io，用 GitHub 帳號登入）：Create app → 選這個 repo、分支 `main`、Main file path 填 `streamlit_app.py`；Advanced settings 選 Python 3.13，Secrets 填 `SHOPPILOT_REPO_URL = "<GitHub repo 網址>"`（側欄才會出現「原始碼與說明（GitHub）」連結）。
4. `streamlit_app.py` 會自動開啟展示模式和只讀快取的 AI，並把示範資料庫複製到暫存資料夾再用；之後推送到 `main`，雲端會自動更新。

## 文件

| 文件 | 內容 |
|---|---|
| [docs/04_已知限制.md](docs/04_已知限制.md) | 資料、分析、競品、AI、系統的已知限制，目前的處理與改善方向 |

## 授權

程式碼以 [MIT License](LICENSE) 授權。

---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- zh
tags:
- lora
- qlora
- tool-calling
- function-calling
- gguf
- llama.cpp
- e-commerce
---

# ShopPilot tools LoRA（Qwen2.5-1.5B-Instruct）

[ShopPilot](https://github.com/kayla162/ShopPilot-demo) 是蝦皮賣家的商品分析工具。這個 LoRA 讓 Qwen2.5-1.5B-Instruct
在「AI 問答」裡**自己選工具查資料再回答**（全店摘要、類別排行、銷量排行、查商品、搜尋商品、側欄選的商品、待處理清單，共 7 個工具），
小模型原本常選錯工具、拿到結果後又說「資料沒有」，或照抄整份清單。

| 檔案 | 用途 |
|---|---|
| `adapter_model.safetensors`、`adapter_config.json` | peft 格式，Hugging Face transformers 載入後合併進模型 |
| `shoppilot-tools-lora-f16.gguf` | llama.cpp 格式，直接套在 Qwen 官方的 Q4_K_M 量化模型上（CPU 就能跑） |
| `train_data.jsonl` | 全部訓練資料（由示範資料產生，沒有真實賣場的資料；system prompt 和工具說明是目前 ShopPilot 的版本，訓練時工具說明裡的範例貨號不同，其他逐字相同） |
| `train_log.json` | 訓練設定和 loss |

## 評估結果

ShopPilot 的自動評估：題目和標準答案都由程式從賣場資料算出（3 個賣場、141 題：示範賣場＋兩家真實賣場，真實賣場的資料不公開），
另外有 30 題手寫的口語題（示範賣場，在最後一版訓練資料定稿之前寫好、只跑一次）。都是 Qwen 官方的 Q4_K_M 量化模型＋llama.cpp（CPU），
只差在有沒有這個 LoRA：

| 做法 | 141 題（95% 信賴區間） | 口語題 |
|---|---|---|
| 工具呼叫 · Qwen2.5-1.5B-Instruct | 83（59%，51–67%） | 16/30 |
| **工具呼叫 · Qwen2.5-1.5B-Instruct ＋ 這個 LoRA** | **135（96%，91–98%）** | **21/30** |
| 工具呼叫 · Qwen3 8B（Ollama，參考） | 133（94%，89–97%） | 25/30 |

- 成對的 McNemar 檢定：微調前後 p < 10⁻¹¹。
- 141 題只有 1 題答錯、5 題回答「資料沒有」；訓練資料只有示範賣場（服飾），兩家真實賣場（服飾、保養品）沒看過，答對 98%、91%。
- 改善最多的題型：問側欄選的商品 0/18 → 18/18、商品銷量 3/15 → 15/15、長尾商品的銷量 0/15 → 15/15。
- 沒學到的：一題問兩件事（「全店銷量最高的是哪個？它的價格是多少？」只答前半）、訓練時沒有的題型（前幾名、某類別的總銷量和商品數）。
  加入這些的訓練資料版本（v2、v3）在 ShopPilot 的 README 有比較；v3 答錯的題目較多是報錯數字，所以發布這一版（v1）。

## 怎麼用

在 ShopPilot 裡（`config/llm.yaml`）：

```yaml
qa_mode: tools
provider: llamacpp
llamacpp:
  lora: models/shoppilot-tools-lora/shoppilot-tools-lora-f16.gguf
```

或直接用 llama-cpp-python：

```python
from llama_cpp import Llama
llm = Llama(model_path="qwen2.5-1.5b-instruct-q4_k_m.gguf", lora_path="shoppilot-tools-lora-f16.gguf", n_ctx=8192)
```

transformers＋peft：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(model, "<這個 repo>").merge_and_unload()
```

工具的定義（JSON schema）在 ShopPilot 的 `shoppilot/llm/qa_tools.py`，system prompt 在 `shoppilot/llm/prompts/shop_qa_tools.md`；
LoRA 是照這兩份訓練的，換別的工具或 prompt 效果會變差。

## 訓練

- 資料：`train_data.jsonl`，662 筆「問題 → 工具呼叫 → 工具回傳 → 回答」的對話，由 ShopPilot 的 `shoppilot/llm/finetune.py`
  從 35 個示範賣場（亂數產生的虛構資料，和評估的賣場不重疊）產生，13 種題型；回答裡的每個數字都出現在工具回傳的結果裡。
- 方法：QLoRA（原本的權重 4-bit NF4、LoRA bfloat16），rank 16、alpha 32、dropout 0.05，q/k/v/o/gate/up/down；
  只算回答的 loss；learning rate 2e-4（cosine）、梯度累積 8、1 個 epoch（75 步）。
- 硬體：RTX 3060 Laptop（6 GB），43.7 分鐘，顯示卡記憶體最高 2.1 GB；驗證 loss 0.52 → 0.004。
- 重現：`python scripts/train_lora.py --qlora --data-version 1`（ShopPilot repo），再用 `scripts/export_lora_gguf.py` 轉成 GGUF。

## 限制

- 只學了 ShopPilot 這 7 個工具和繁體中文的賣場問題；不是通用的工具呼叫模型。
- 題目和標準答案由程式產生，評分只檢查回答裡有沒有標準答案（商品、類別、數字），不檢查其他句子。
- 回答裡的數字由 ShopPilot 的數字檢查把關（數字一定要出現在工具回傳的結果裡，否則不顯示）；單獨使用這個 LoRA 時沒有這層保護。

## 授權

LoRA 權重和原本的模型一樣用 Apache 2.0；ShopPilot 的程式碼是 MIT。

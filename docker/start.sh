#!/bin/sh
# 示範資料庫複製到可寫的暫存位置：訪客自己問的問題會寫進快取，容器重啟後清掉；預先跑好的答案在映像裡
set -e
mkdir -p "$SHOPPILOT_DATA_DIR"
cp demo/shoppilot.db "$SHOPPILOT_DATA_DIR/"
# 用 --skip-ai 建的示範資料沒有 examples.json：照樣啟動，只是問答分頁沒有預先算好的範例問題
if [ -f demo/examples.json ]; then cp demo/examples.json "$SHOPPILOT_DATA_DIR/"; fi
# 專案的 .streamlit/config.toml 只開在本機；容器裡要對外開放 7860 埠
exec streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true --client.toolbarMode minimal

# ShopPilot 完整版（可以即時問本機 AI）：本機 Docker；有 Hugging Face PRO 的話也能直接當 Docker Space 部署
FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU 版 torch 比預設版本小很多；之後裝 requirements-local.txt 時 torch 已經滿足，不會換成 CUDA 版
COPY --chown=user requirements.txt requirements-local.txt ./
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user -r requirements.txt -r requirements-local.txt

# 建置時就下載模型，執行時完全不連網
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"

COPY --chown=user . .
ENV SHOPPILOT_DEMO=1 SHOPPILOT_DATA_DIR=/tmp/shoppilot HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 7860
CMD ["sh", "docker/start.sh"]

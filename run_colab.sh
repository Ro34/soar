#!/bin/bash
set -euo pipefail

# 日志输出到文件（可通过环境变量覆盖）
WORKSPACE_ROOT="/content/soar"
LOG_DIR="${LOG_DIR:-${WORKSPACE_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_colab_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "日志写入：${LOG_FILE}"

# ==========================================
# SOAR 2026: MiniCPM-SALA Colab 一站式启动脚本
# ==========================================

# ---------------------------
# 路径配置区 (适配 Colab)
# ---------------------------
WORKSPACE_ROOT="/content/soar"
REPO_DIR="${WORKSPACE_ROOT}/sglang"

# 强烈建议在 Colab 中挂载 Google Drive 并将模型存入 Drive
# 否则每次断开连接都需要重新下载十几 GB 的权重！
# 请在运行前先在 Notebook 中执行:
# from google.colab import drive; drive.mount('/content/drive')
DEFAULT_MODEL_PATH="/content/drive/MyDrive/models/MiniCPM-SALA"
MODEL_REPO="openbmb/MiniCPM-SALA"
export MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

# 启动参数
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-true}"

SETUP_ONLY=false

gpu_snapshot() {
    echo -e "\n[GPU] $1"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || echo "⚠️ nvidia-smi 执行失败。"
    echo
}

# ---------------------------
# 1. 基础依赖安装 (Colab 简化版)
# ---------------------------
echo "[1/4] 检查并安装 Colab 基础依赖..."
# Colab 自带 pip，不需要安装 pipx。直接安装必须的系统工具
apt-get update -qq && apt-get install -y -qq psmisc  # 提供 fuser 命令，用于清理端口占用
pip install -q gpustat uv

# Colab 默认 CUDA 路径
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:$PATH"

if ! command -v nvcc &> /dev/null; then
    echo "❌ 错误: Colab 环境中未找到 nvcc，请检查当前运行时是否为 GPU (T4/L4/A100) 模式。"
    exit 1
fi

# ---------------------------
# 2. 拉取 SGLang 代码
# ---------------------------
echo -e "\n[2/4] 获取 SGLang (minicpm_sala 分支)..."
if [ ! -d "$REPO_DIR" ]; then
    cd "$WORKSPACE_ROOT"
    git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
else
    echo "✅ 目录 '$REPO_DIR' 已存在，跳过克隆。"
fi

cd "$REPO_DIR" || exit 1

# ---------------------------
# 3. 环境配置与 CUDA 扩展编译
# ---------------------------
echo -e "\n[3/4] 检查环境与编译状态..."
# 这里不再判断 VENV_DIR，而是通过检查 sglang 是否已安装在系统环境中来判断
if ! python -c "import sglang" &> /dev/null; then
    echo "检测到尚未安装 sglang 环境。"
    
    # 【非常重要】这里调用的必须是我们上一轮修改好的 "Colab 专用版" 编译脚本！
    # 假设你将上一份脚本保存为了 install_minicpm_sala_colab.sh
    if [ -f "${WORKSPACE_ROOT}/install_minicpm_sala_colab.sh" ]; then
        bash "${WORKSPACE_ROOT}/install_minicpm_sala_colab.sh"
    else
        echo "❌ 错误: 找不到 Colab 专用编译脚本 install_minicpm_sala_colab.sh"
        echo "请将上一轮我为你写的脚本上传到 /content 目录下并重试。"
        exit 1
    fi
else
    echo "✅ 检测到 sglang 已安装到系统环境，跳过重新编译。"
fi

if [ "$SETUP_ONLY" = true ]; then
    echo "SETUP_ONLY=true，环境配置完毕，退出脚本。"
    exit 0
fi

# ---------------------------
# 4. 启动推理服务
# ---------------------------
echo -e "\n[4/4] 准备启动 Baseline 推理服务..."

# 确保运行时依赖是最新的（使用 --system 安装到全局）
uv pip install --system -U "huggingface_hub[cli]>=0.34,<1.0" hf_transfer

# 处理模型下载逻辑
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "未检测到完整模型权重，准备下载 ${MODEL_REPO} 到 ${MODEL_PATH}..."
    mkdir -p "$(dirname "${MODEL_PATH}")"
    
    # 提醒配置 HuggingFace Token
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "⚠️ 注意: 你没有设置 HF_TOKEN 环境变量。如果该模型是受限访问的，下载将会失败。"
        echo "建议在 Colab Notebook 中设置: import os; os.environ['HF_TOKEN'] = '你的Token'"
    fi

    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --resume-download "${MODEL_REPO}" --local-dir "${MODEL_PATH}"
fi

echo "=========================================="
echo "正在启动 SGLang 推理框架 (MiniCPM-SALA 模式)..."
echo "模型路径: $MODEL_PATH"
echo "赛题规则: 禁用 radix-cache"
echo "配置: mem_fraction_static=$MEM_FRACTION_STATIC, kv_cache_dtype=$KV_CACHE_DTYPE"
echo "调度: max_running_requests=$MAX_RUNNING_REQUESTS, chunk_size=$CHUNKED_PREFILL_SIZE"
echo "=========================================="

# 清理残留进程 (释放显存和端口)
echo "清理残留的 sglang 进程..."
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang_router" 2>/dev/null || true
fuser -k 31111/tcp 2>/dev/null || true
sleep 2

gpu_snapshot "启动 SGLang 前最终检查"

launch_cmd=(
    python3 -m sglang.launch_server
    --model "${MODEL_PATH}"
    --trust-remote-code
    --disable-radix-cache
    --attention-backend minicpm_flashinfer
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --skip-server-warmup
    --port 31111
    --dense-as-sparse
)

if [ "${DISABLE_CUDA_GRAPH}" = "true" ]; then
    launch_cmd+=(--disable-cuda-graph)
fi


echo "启动命令: ${launch_cmd[*]}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_MODULE_LOADING=LAZY "${launch_cmd[@]}"
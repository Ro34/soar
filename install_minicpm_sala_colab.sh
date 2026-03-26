#!/bin/bash
set -e

# Colab 环境下硬编码 sglang 仓库目录
REPO_ROOT="/content/soar/sglang"
DEPS_DIR="${REPO_ROOT}/3rdparty"

export UV_INDEX_URL="https://pypi.org/simple"

echo "============================================"
echo " MiniCPM-SALA Colab Installation (Clean Version)"
echo "============================================"

# 确保 uv 存在
if ! command -v uv &> /dev/null; then
    pip install -q uv
fi

cd "${REPO_ROOT}"
git submodule update --init --recursive

# Colab 标准环境 CUDA 配置
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:$PATH"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

# 强制单线程编译，防止 OOM (直接写在脚本里)
export MAX_JOBS=1

echo "[1/4] Installing sglang..."
uv pip install --system "cmake>=3.26"
uv pip install --system --upgrade pip setuptools wheel
uv pip install --system -e "${REPO_ROOT}/python"

# 动态获取 Python 包路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
NVIDIA_PKG_ROOT="${SITE_PACKAGES}/nvidia"

echo "[2/4] Building CUDA kernels..."
cd "${DEPS_DIR}/infllmv2_cuda_impl"
git submodule update --init --recursive

# 核心修改：移除全局扫描，仅精准注入 4 个必需的核心代数库，避开 cccl 冲突
if [ -d "${NVIDIA_PKG_ROOT}" ]; then
    for pkg in cusparse cublas cusolver cusparselt; do
        include_dir="${NVIDIA_PKG_ROOT}/${pkg}/include"
        lib_dir="${NVIDIA_PKG_ROOT}/${pkg}/lib"

        if [ -d "${include_dir}" ]; then
            export CPATH="${include_dir}${CPATH:+:$CPATH}"
        fi
        if [ -d "${lib_dir}" ]; then
            export LIBRARY_PATH="${lib_dir}${LIBRARY_PATH:+:$LIBRARY_PATH}"
            export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    done
fi

echo "  - Installing infllm_v2..."
python setup.py install

echo "  - Installing sparse_kernel..."
cd "${DEPS_DIR}/sparse_kernel"
python setup.py install

echo "[4/4] Installing additional libraries..."
uv pip install --system tilelang flash-linear-attention

echo "============================================"
echo " Installation complete for Colab!"
echo "============================================"
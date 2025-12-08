# Optimized Dockerfile - HPC + Backend Stack (based on nvidia/cuda:12.4.0-devel-ubuntu22.04)
# Multi-stage: add a PyTorch builder stage to compile a wheel for MX250 (sm_61),
# then keep your original final stage content (unchanged except to install the built wheel).
# Build-time flags:
#  --build-arg BUILD_PYTORCH=true|false
#  --build-arg TORCH_CUDA_ARCH_LIST="6.1"
#  --build-arg PYTORCH_TAG=v2.6.0
#  --build-arg CUDA_VERSION=12.4.0
#  --build-arg MAX_JOBS=4
ARG BUILD_PYTORCH=true
ARG TORCH_CUDA_ARCH_LIST="6.1"
ARG PYTORCH_TAG=v2.6.0
ARG CUDA_VERSION=12.4.0
ARG MAX_JOBS=4

# --------------------------
# Stage 0: PyTorch builder
# --------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS pytorch-builder

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    CMAKE_C_COMPILER=/usr/bin/gcc-12 \
    CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    PYTORCH_BUILD_VERSION=${PYTORCH_TAG#v}

WORKDIR /tmp

# Minimal build deps for PyTorch; keep it focussed to reduce image size
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build ccache python3 python3-dev python3-pip \
      python3-setuptools python3-wheel patchelf pkg-config libnuma-dev wget curl \
      gcc-12 g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# Clone and (conditionally) build PyTorch wheel
ARG BUILD_PYTORCH
ARG PYTORCH_TAG
ARG MAX_JOBS
RUN set -eux; \
    if [ "${BUILD_PYTORCH}" = "true" ]; then \
      git clone --recursive --depth 1 --branch ${PYTORCH_TAG} https://github.com/pytorch/pytorch.git pytorch-src; \
      cd pytorch-src; \
      python3 -m pip install -r requirements.txt || true; \
      # determine a sensible parallelism value and export it as MAX_JOBS for the PyTorch build system
      NJOBS="$(nproc 2>/dev/null || echo ${MAX_JOBS})"; \
      echo "Using ${NJOBS} parallel jobs for PyTorch build (MAX_JOBS=${NJOBS})"; \
      # setuptools bdist_wheel does not accept -j; PyTorch build uses MAX_JOBS env var
      MAX_JOBS=${NJOBS} python3 setup.py bdist_wheel || (echo "first attempt failed, retry with MAX_JOBS=4" && MAX_JOBS=4 python3 setup.py bdist_wheel); \
      mkdir -p /tmp/out && mv dist/*.whl /tmp/out/pytorch.whl; \
    else \
      # create an empty placeholder so COPY in final stage never fails
      mkdir -p /tmp/out && touch /tmp/out/pytorch.whl; \
    fi

# Ensure the produced wheel (or placeholder) is at /tmp/out/pytorch.whl
RUN ls -l /tmp/out || true

# --------------------------
# Final stage: original image base + your full content
# --------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

WORKDIR /workspace

# Basic deps + PPAs (ensure bzip2/tar for .tar.bz2 extraction and tools for verification)
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      lsb-release \
      wget \
      curl \
      ca-certificates \
      gnupg2 \
      apt-transport-https \
      dirmngr \
      bzip2 \
      tar \
      file \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      # compilers & build tools
      gcc-12 g++-12 gfortran-12 build-essential cmake ninja-build make automake autoconf libtool pkg-config \
      # development tools
      git vim nano sudo openssh-client openssh-server \
      # debugging / profiling
      gdb gdbserver lldb valgrind cppcheck splint flawfinder strace ltrace htop iotop sysstat stress-ng \
      # MPI
      libopenmpi-dev openmpi-bin openmpi-common mpi-default-bin mpi-default-dev \
      # scientific libs / testing / db clients
      libfftw3-dev libblas-dev liblapack-dev libopenblas-dev libhdf5-dev libnetcdf-dev \
      libgtest-dev google-mock libcppunit-dev libboost-all-dev \
      postgresql-client mysql-client redis-tools jq zip unzip sqlite3 sysbench \
      # other tooling (may be replaced by official repos if needed)
      docker-compose \
      hwloc slurm-client \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Robust micromamba install: download, extract, detect binary location, ensure exec bit and symlink
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN set -eux; \
    MICROMAMBA_URL="https://micromamba.snakepit.net/api/micromamba/linux-64/latest"; \
    mkdir -p /opt/micromamba /opt/micromamba/bin /tmp/micromamba-extract; \
    wget -O /tmp/micromamba.tar.bz2 "${MICROMAMBA_URL}"; \
    if [ ! -s /tmp/micromamba.tar.bz2 ]; then echo "micromamba archive is empty"; ls -l /tmp/micromamba.tar.bz2; exit 1; fi; \
    tar -xjf /tmp/micromamba.tar.bz2 -C /tmp/micromamba-extract || { echo "tar extraction failed"; ls -la /tmp/micromamba-extract; exit 1; }; \
    # Find micromamba binary in the extracted tree and move to /opt/micromamba/bin
    if [ -f /tmp/micromamba-extract/bin/micromamba ]; then \
      mv /tmp/micromamba-extract/bin/* /opt/micromamba/bin/; \
    elif [ -f /tmp/micromamba-extract/micromamba ]; then \
      mv /tmp/micromamba-extract/micromamba /opt/micromamba/bin/; \
    else \
      # fallback: search for any file named micromamba and move it
      FOUND="$(find /tmp/micromamba-extract -type f -name micromamba | head -n1)"; \
      if [ -n "$FOUND" ]; then mv "$FOUND" /opt/micromamba/bin/; else echo "micromamba binary not found in archive"; ls -R /tmp/micromamba-extract; exit 1; fi; \
    fi; \
    chmod -R a+rx /opt/micromamba /opt/micromamba/bin || true; \
    ln -sf /opt/micromamba/bin/micromamba /usr/local/bin/micromamba; \
    rm -rf /tmp/micromamba.tar.bz2 /tmp/micromamba-extract; \
    # verify executable and print diagnostics on failure
    if ! /usr/local/bin/micromamba --version >/dev/null 2>&1; then \
      echo "=== micromamba diagnostics ==="; ls -la /opt/micromamba || true; ls -la /opt/micromamba/bin || true; file /opt/micromamba/bin/micromamba || true; ldd /opt/micromamba/bin/micromamba || true; /usr/local/bin/micromamba --version || true; echo "=== end diagnostics ==="; exit 1; fi

# Ensure micromamba and (eventual) env come first on PATH for subsequent RUNs
ENV PATH="/opt/micromamba/bin:${MAMBA_ROOT_PREFIX}/env/bin:${PATH}"

# ARG flags for optional AI/LLM/Quantum layers
ARG ENABLE_AI=true
ARG ENABLE_LLM=true
ARG ENABLE_QUANTUM=true
ARG ENABLE_TPU=false
ARG ENABLE_NPU=false

# Keep CUDA_VERSION arg in the final stage for reference
ARG CUDA_VERSION

# Create a named micromamba env (use absolute micromamba path to avoid PATH timing issues)
RUN if [ "${ENABLE_AI}" = "true" ] || [ "${ENABLE_LLM}" = "true" ] || [ "${ENABLE_QUANTUM}" = "true" ]; then \
      /opt/micromamba/bin/micromamba create -y -p ${MAMBA_ROOT_PREFIX}/env python=3.11 && \
      /opt/micromamba/bin/micromamba clean --all -f -y; \
    fi

# From here, prefer the micromamba env python when available
ENV PATH="${MAMBA_ROOT_PREFIX}/env/bin:${PATH}"

# Install GitLab Runner from official packages.gitlab.com repository (signed)
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends gnupg curl lsb-release ca-certificates && \
    curl -fsSL https://packages.gitlab.com/runner/gitlab-runner/gpgkey | gpg --dearmor -o /usr/share/keyrings/gitlab-runner-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/gitlab-runner-archive-keyring.gpg] https://packages.gitlab.com/runner/gitlab-runner/ubuntu/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/gitlab-runner.list && \
    apt-get update --fix-missing && apt-get install -y --no-install-recommends gitlab-runner && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install LLVM 17 via the LLVM script (non-interactive).
RUN wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh && \
    chmod +x /tmp/llvm.sh && \
    /tmp/llvm.sh 17 && \
    rm /tmp/llvm.sh

# Install Python dependencies (use micromamba's python when available; fallback to system)
RUN set -eux; \
    if command -v python3 >/dev/null 2>&1; then \
      python3 -m pip install --upgrade pip setuptools wheel || true; \
    fi; \
    python3 -m pip install --no-cache-dir \
      cuda-python \
      cupy-cuda12x \
      pycuda \
      numba \
      numpy \
      scipy \
      scikit-learn \
      pandas \
      matplotlib \
      jupyter \
      xarray \
      "dask[complete]" \
      mpi4py \
      netCDF4 \
      h5py \
      pytest tox || true

# Install Go 1.24
RUN wget -q https://go.dev/dl/go1.24.0.linux-amd64.tar.gz -O /tmp/go.tar.gz && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && rm /tmp/go.tar.gz
ENV GOROOT=/usr/local/go \
    GOPATH=/root/go \
    PATH=$PATH:/usr/local/go/bin:/root/go/bin

# Install Rust (non-interactive)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    /root/.cargo/bin/rustc --version
ENV PATH="/root/.cargo/bin:${PATH}"

# Build NVIDIA CCCL (keep separate so failures are isolated)
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends git cmake build-essential && \
    cd /tmp && git clone --depth 1 https://github.com/NVIDIA/cccl.git && \
    cd cccl && mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CXX_COMPILER=g++-12 \
      -DCMAKE_C_COMPILER=gcc-12 \
      -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" && make install && \
    cd / && rm -rf /tmp/cccl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: NVIDIA extras (cuDNN/NCCL placeholders)
RUN if [ "${ENABLE_AI}" = "true" ]; then \
      apt-get update && \
      # attempt to install libcudnn / nccl packages if present in distro repos (best-effort)
      apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev libnccl2 libnccl-dev || true && \
      apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# --------------------------
# Copy PyTorch wheel from builder stage (if built)
# --------------------------
# Copy the wheel (or placeholder) produced by the builder stage into the final image
COPY --from=pytorch-builder /tmp/out/pytorch.whl /tmp/pytorch.whl

# ML/LLM/Quantum Python layers (respect ARG flags)
RUN if [ "${ENABLE_AI}" = "true" ]; then \
      python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
      # If builder produced a real wheel it will be installed; otherwise fallback to pip wheel (cu124 index)
      python3 -m pip install --no-cache-dir /tmp/pytorch.whl || \
        pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 || true && \
      # TensorFlow: try GPU wheel for recent TF first, fallback to CPU-only if incompatible
      pip install --no-cache-dir tensorflow==2.20.* || pip install --no-cache-dir tensorflow-cpu || true && \
      # JAX: use the CUDA12 plugin when available (ML-OSS index). This targets CUDA12-compatible jaxlib.
      pip install -U --pre jax jaxlib "jax-cuda12-plugin[with-cuda]" jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ || true && \
      /opt/micromamba/bin/micromamba clean --all -f -y || true; \
    fi && \
    if [ "${ENABLE_LLM}" = "true" ]; then \
      pip install --no-cache-dir transformers accelerate optimum datasets tokenizers safetensors || true && \
      pip install --no-cache-dir deepspeed || true && \
      pip install --no-cache-dir bitsandbytes || true && \
      # xformers may require a compiled wheel for your CUDA/torch combo; attempt pip first
      pip install --no-cache-dir xformers || true; \
    fi && \
    if [ "${ENABLE_QUANTUM}" = "true" ]; then \
      pip install --no-cache-dir qiskit==0.47.* cirq pennylane qulacs || true && \
      pip install --no-cache-dir cuquantum || true; \
    fi

# Helpful CI / dev tools for LLM training & profiling
RUN pip install --no-cache-dir nvidia-ml-py3 gpustat nvtx-plugins || true && \
    apt-get update && apt-get install -y --no-install-recommends htop jq && rm -rf /var/lib/apt/lists/*

# Final housekeeping
RUN echo "AI/LLM/Quantum layers added (flags: AI=${ENABLE_AI} LLM=${ENABLE_LLM} QUANTUM=${ENABLE_QUANTUM} TPU=${ENABLE_TPU} NPU=${ENABLE_NPU})"

# Workspace and misc
RUN mkdir -p /workspace && echo "✅ Base system ready"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD gcc --version > /dev/null 2>&1

CMD ["/bin/bash"]
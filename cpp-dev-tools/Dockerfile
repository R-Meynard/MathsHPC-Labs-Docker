# Optimized Dockerfile - HPC + Backend Stack (based on nvidia/cuda:12.1.0-devel-ubuntu22.04)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

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

# Robust micromamba install: download, extract, chmod +x, verify binary and show diagnostics on failure
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN set -eux; \
    MICROMAMBA_URL="https://micromamba.snakepit.net/api/micromamba/linux-64/latest"; \
    mkdir -p /opt/micromamba; \
    wget -O /tmp/micromamba.tar.bz2 "${MICROMAMBA_URL}"; \
    # Basic sanity check: file size > 0
    if [ ! -s /tmp/micromamba.tar.bz2 ]; then echo "micromamba archive is empty"; ls -l /tmp/micromamba.tar.bz2; exit 1; fi; \
    tar -xjf /tmp/micromamba.tar.bz2 -C /opt/micromamba --strip-components=1; \
    # ensure executables have exec bit (some tar variants might preserve, enforce it)
    chmod -R a+rx /opt/micromamba/bin || true; \
    ln -sf /opt/micromamba/bin/micromamba /usr/local/bin/micromamba; \
    rm -f /tmp/micromamba.tar.bz2; \
    # Verify: if micromamba fails, dump diagnostics to help debugging
    if ! /usr/local/bin/micromamba --version >/dev/null 2>&1; then \
      echo "=== micromamba diagnostics ==="; \
      ls -la /opt/micromamba || true; \
      ls -la /opt/micromamba/bin || true; \
      file /opt/micromamba/bin/micromamba || true; \
      ldd /opt/micromamba/bin/micromamba || true; \
      /usr/local/bin/micromamba --version || true; \
      echo "=== end diagnostics ==="; \
      exit 1; \
    fi

# Ensure micromamba and (eventual) env come first on PATH for subsequent RUNs
ENV PATH="/opt/micromamba/bin:${MAMBA_ROOT_PREFIX}/env/bin:${PATH}"

# ARG flags for optional AI/LLM/Quantum layers
ARG ENABLE_AI=true
ARG ENABLE_LLM=true
ARG ENABLE_QUANTUM=true
ARG ENABLE_TPU=false
ARG ENABLE_NPU=false
ARG CUDA_VERSION=12.1

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
      apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev libnccl2 libnccl-dev || true && \
      apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# ML/LLM/Quantum Python layers (respect ARG flags)
RUN if [ "${ENABLE_AI}" = "true" ]; then \
      pip install --no-cache-dir -U pip setuptools wheel && \
      pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 || true && \
      pip install --no-cache-dir tensorflow==2.14.* || true && \
      pip install --no-cache-dir jax jaxlib==0.4.*+cuda121 -f https://storage.googleapis.com/jax-releases/jax_releases.html || true && \
      /opt/micromamba/bin/micromamba clean --all -f -y || true; \
    fi && \
    if [ "${ENABLE_LLM}" = "true" ]; then \
      pip install --no-cache-dir transformers accelerate optimum datasets tokenizers safetensors || true && \
      pip install --no-cache-dir deepspeed || true && \
      pip install --no-cache-dir bitsandbytes || true && \
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
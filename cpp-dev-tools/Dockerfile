# Optimized Dockerfile - HPC + Backend Stack (based on nvidia/cuda:12.1.0-devel-ubuntu22.04)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

WORKDIR /workspace

# Combine package installs where reasonable to reduce layers.
# Add PPAs needed (toolchain for newer gcc, deadsnakes for python3.11).
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
      gitlab-runner \
      hwloc slurm-client \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install LLVM 17 via the LLVM script (non-interactive).
RUN wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh && \
    chmod +x /tmp/llvm.sh && \
    /tmp/llvm.sh 17 && \
    rm /tmp/llvm.sh

# Install Python 3.11 and pip packages using python3.11 explicitly.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Go 1.24
RUN wget -q https://go.dev/dl/go1.24.0.linux-amd64.tar.gz -O /tmp/go.tar.gz && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && rm /tmp/go.tar.gz
ENV GOROOT=/usr/local/go \
    GOPATH=/root/go \
    PATH=$PATH:/usr/local/go/bin:/root/go/bin

# Install Rust (non-interactive), verify in same RUN and set PATH via ENV
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

# Python packages: group heavy installs to reduce layers. Use python3 -m pip to ensure correct binary.
# If you need reproducible builds, pin versions or use constraints files.
RUN python3 -m pip install --upgrade \
      pip setuptools wheel && \
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
      pytest tox


# Extensions pour AI / LLM / Quantum (snippet à intégrer dans votre Dockerfile existant)
ARG ENABLE_AI=true
ARG ENABLE_LLM=true
ARG ENABLE_QUANTUM=true
ARG ENABLE_TPU=false    # TPU require cloud/runtime - leave false for local builds
ARG ENABLE_NPU=false    # NPU (Graphcore/Habana/Ascend) - vendor images preferred
ARG CUDA_VERSION=12.1

# -- prerequisites: micromamba (recommended) to manage Python & binaries --
RUN set -eux; \
    if [ "${ENABLE_AI}" = "true" ] || [ "${ENABLE_LLM}" = "true" ] || [ "${ENABLE_QUANTUM}" = "true" ]; then \
      wget -qO /tmp/micromamba.tar.bz2 "https://micromamba.snakepit.net/api/micromamba/linux-64/latest" && \
      mkdir -p /opt/micromamba && \
      tar -xjf /tmp/micromamba.tar.bz2 -C /opt/micromamba --strip-components=1 && \
      ln -s /opt/micromamba/bin/micromamba /usr/local/bin/micromamba && \
      rm /tmp/micromamba.tar.bz2; \
    fi

# -- Create a named environment to keep Python deps isolated --
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN if [ "${ENABLE_AI}" = "true" ] || [ "${ENABLE_LLM}" = "true" ] || [ "${ENABLE_QUANTUM}" = "true" ]; then \
      micromamba create -y -p ${MAMBA_ROOT_PREFIX}/env python=3.11 && \
      micromamba clean --all -f -y; \
    fi
ENV PATH="${MAMBA_ROOT_PREFIX}/env/bin:${PATH}"

# -- NVIDIA extras (cuDNN, NCCL, TensorRT) - prefer NVIDIA apt repos or use NGC containers --
# NOTE: on many systems these should come from the host / vendor image; here we show apt-style install placeholders.
RUN if [ "${ENABLE_AI}" = "true" ]; then \
      apt-get update && \
      # Example: install cuDNN / NCCL packages (names and availability depend on CUDA version & NVIDIA repo)
      # You should add NVIDIA's apt repo and keys prior to these installs for correct versions.
      apt-get install -y --no-install-recommends \
        libcudnn8 libcudnn8-dev libnccl2 libnccl-dev && \
      apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# -- Install common ML & LLM Python packages --
RUN if [ "${ENABLE_AI}" = "true" ]; then \
      # install core ML libs - prefer pinned versions for reproducibility
      pip install --no-cache-dir -U pip setuptools wheel && \
      # PyTorch (choose correct wheel or use conda channel if wheel not available)
      pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 || true && \
      # TensorFlow (ensure TF build matches CUDA / cuDNN versions); may require specific package or build
      pip install --no-cache-dir tensorflow==2.14.* || true && \
      # JAX + jaxlib for CUDA (check jaxlib wheel for CUDA 12.x)
      pip install --no-cache-dir jax jaxlib==0.4.*+cuda121 -f https://storage.googleapis.com/jax-releases/jax_releases.html || true && \
      # other useful libs
      pip install --no-cache-dir numpy scipy pandas scikit-learn matplotlib numba mpi4py && \
      micromamba clean --all -f -y; \
    fi

# -- LLM ecosystem (huggingface, accelerate, deepspeed, bitsandbytes, xformers, transformers) --
RUN if [ "${ENABLE_LLM}" = "true" ]; then \
      pip install --no-cache-dir transformers accelerate optimum datasets tokenizers safetensors && \
      # DeepSpeed (may require compiling C++/CUDA extensions) - prefer installing from released wheel
      pip install --no-cache-dir deepspeed && \
      # bitsandbytes: requires CUDA compatibility (may fail on unsupported CUDA)
      pip install --no-cache-dir bitsandbytes || true && \
      # xformers: often needs to be built from source for your CUDA/GCC combo; provide placeholder
      pip install --no-cache-dir xformers || true; \
    fi

# -- Quantum SDKs and simulators --
RUN if [ "${ENABLE_QUANTUM}" = "true" ]; then \
      pip install --no-cache-dir qiskit==0.47.* cirq pennylane qulacs || true && \
      # NVIDIA cuQuantum (pip package) for GPU accelerated simulation (requires CUDA libs)
      pip install --no-cache-dir cuquantum || true; \
    fi

# -- TPU note: libtpu / JAX on TPU require cloud TPU VM or specific packages --
# If ENABLE_TPU=true then user must swap to official Google Cloud TPU VM images or install libtpu
# (not shown here; see https://cloud.google.com/tpu/docs)

# -- NPU note: for Graphcore/Habana/Ascend use vendor base images & SDKs --
# If ENABLE_NPU=true, fetch vendor images (Graphcore Poplar, Habana SynapseAI) and follow vendor docs.

# -- Helpful CI / dev tools for LLM training & profiling --
RUN pip install --no-cache-dir nvidia-ml-py3 gpustat nvtx-plugins && \
    apt-get update && apt-get install -y --no-install-recommends htop jq && rm -rf /var/lib/apt/lists/*

# Final housekeeping
RUN echo "AI/LLM/Quantum layers added (flags: AI=${ENABLE_AI} LLM=${ENABLE_LLM} QUANTUM=${ENABLE_QUANTUM} TPU=${ENABLE_TPU} NPU=${ENABLE_NPU})"

# Workspace and misc
RUN mkdir -p /workspace && echo "✅ Base system ready"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD gcc --version > /dev/null 2>&1

CMD ["/bin/bash"]
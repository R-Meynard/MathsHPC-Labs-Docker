# ============================================================================
# OPTIMIZED DOCKERFILE - HPC + Backend Stack
# ============================================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

# ========== Fix APT sources with REAL mirrors ==========
RUN sed -i 's|http://archive.ubuntu.com|http://archive.ubuntu.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://security.ubuntu.com|g' /etc/apt/sources.list

# ========== Update repositories ==========
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Install base tools first ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    gnupg2 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Add PPA for GCC 13 ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    lsb-release && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    gfortran-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Build essentials ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    make \
    automake \
    autoconf \
    libtool \
    pkg-config \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Development tools ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    nano \
    sudo \
    openssh-server \
    openssh-client \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Monitoring & profiling ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    htop \
    iotop \
    sysstat \
    stress-ng \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== MPI ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    mpi-default-bin \
    mpi-default-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Clang/LLVM (from LLVM repo) ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    lsb-release \
    wget && \
    wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh && \
    chmod +x /tmp/llvm.sh && \
    /tmp/llvm.sh 17 && \
    rm /tmp/llvm.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Debugging tools ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    gdb \
    gdbserver \
    lldb \
    valgrind \
    cppcheck \
    splint \
    flawfinder \
    strace \
    ltrace \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Testing frameworks ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgtest-dev \
    google-mock \
    libcppunit-dev \
    libboost-all-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Scientific libraries ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libhdf5-dev \
    libnetcdf-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Database tools ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    mysql-client \
    redis-tools \
    jq \
    zip \
    unzip \
    sqlite3 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Python 3.11 ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 && \
    python3.11 -m pip install --upgrade pip setuptools wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Node.js & npm ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Sysbench ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    sysbench \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== CUDA Nsight Systems ==========
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    cuda-nsight-systems-12-1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Install CCCL (CUDA C++ Core Libraries) ==========
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/cccl.git && \
    cd cccl && \
    mkdir build && cd build && \
    cmake .. \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/cccl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Install CUDA Python ==========
RUN python3 -m pip install --upgrade \
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
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ========== Go 1.24.0 ==========
RUN wget -q https://go.dev/dl/go1.24.0.linux-amd64.tar.gz -O /tmp/go.tar.gz && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && \
    rm /tmp/go.tar.gz

ENV GOROOT=/usr/local/go \
    GOPATH=/root/go \
    PATH=$PATH:/usr/local/go/bin:/root/go/bin

# ========== Rust ==========
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env

ENV PATH=$PATH:/root/.cargo/bin

# ========== Workspace setup ==========
RUN mkdir -p /workspace && \
    echo "✅ Base system ready"

WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD gcc --version > /dev/null 2>&1

CMD ["/bin/bash"]
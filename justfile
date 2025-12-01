build: build-rocksdb build-fio build-llama build-micro

bench: bench-ann bench-rocksdb bench-llama bench-fio
bench-repl: bench-ann-repl bench-rocksdb-repl bench-llama-repl bench-fio-repl

plot: plot-ann plot-rocksdb plot-fio

build-rocksdb:
    #!/usr/bin/env bash
    set -euxo pipefail
    mkdir -p rocksdb/build
    cd rocksdb/build
    cmake -DCMAKE_BUILD_TYPE=Release -DFAIL_ON_WARNINGS=OFF -DWITH_ZSTD=ON ..
    make -j

build-micro:
    make -C microbench -j

build-fio:
    make -C fio-3.40 -j

build-llama:
    #!/usr/bin/env bash
    cd llama.cpp

    set -euxo pipefail
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
    apt update

    apt-get install -y intel-oneapi-mkl-devel
    apt-get install -y intel-oneapi-compiler-dpcpp-cpp
    apt install -y pkg-config
    apt-get install -y libcurl4-openssl-dev

    [ ! -f "Llama-3.1-Tulu-3-8B-Q8_0.gguf" ] && wget "https://huggingface.co/lmstudio-community/Llama-3.1-Tulu-3-8B-GGUF/resolve/7033c16b4f79f8708a27d80bf2ae0c6537253d1b/Llama-3.1-Tulu-3-8B-Q8_0.gguf?download=true" -O Llama-3.1-Tulu-3-8B-Q8_0.gguf

    set +u
    . /opt/intel/oneapi/setvars.sh
    set -u

    cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
    cmake --build build --config Release -- -j$(nproc)

bench-rocksdb:
    uv run run.py rocksdb

bench-rocksdb-repl:
    uv run run.py rocksdb-repl

bench-ann:
    uv run run.py ann

bench-ann-repl:
    uv run run.py ann-repl

bench-fio:
    uv run run.py fio

bench-fio-repl:
    uv run run.py fio-repl

bench-llama:
    #!/usr/bin/env bash
    . /opt/intel/oneapi/setvars.sh
    uv run run.py llama

bench-llama-repl:
    #!/usr/bin/env bash
    . /opt/intel/oneapi/setvars.sh
    uv run run.py llama-repl

plot-ann:
    uv run run.py plot-ann

plot-rocksdb:
    uv run run.py plot-rocksdb

plot-fio:
    uv run run.py plot-fio

plot-monitoring:
    uv run run.py plot-monitoring

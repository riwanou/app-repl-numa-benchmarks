build: build-rocksdb build-fio build-llama build-micro build-dirtest

bench: bench-ann bench-pressure bench-rocksdb bench-llama bench-fio bench-pgtable-sync-baseline
bench-repl: bench-ann-repl bench-pressure-repl bench-rocksdb-repl bench-llama-repl bench-fio-repl bench-locality-repl bench-pgtable-sync-repl

plot: plot-ann plot-rocksdb plot-llama plot-fio plot-fio-pgtable plot-pressure plot-sharing

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

build-dirtest:
    make -C dirtest

build-ann:
    uv run run.py build-ann

build-duckdb:
    uv run run.py build-duckdb

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
    cmake --build build --config Release --target llama-bench -- -j$(nproc)

bench-rocksdb:
    uv run run.py rocksdb

bench-rocksdb-repl:
    uv run run.py rocksdb-repl

bench-ann:
    uv run run.py ann

bench-ann-repl:
    uv run run.py ann-repl

bench-locality-repl:
    uv run run.py locality

bench-pressure:
    uv run run.py pressure

bench-pressure-repl:
    uv run run.py pressure-repl

# fio read penalty from Intel MLC: sequential read bandwidth on every hw
# thread, each on local memory vs one interleaved copy, P = 100 (local / il - 1)
mlc-penalty:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -x mlc/Linux/mlc ]; then
        mkdir -p mlc
        curl -sL https://downloadmirror.intel.com/926327/mlc_v3.13.tgz -o mlc/mlc.tgz
        echo "a8537e8ff3fad626d75a383fabc224ccc4cc98a0111c9989f7fb26b639f12019  mlc/mlc.tgz" | sha256sum -c --quiet
        tar xzf mlc/mlc.tgz -C mlc Linux/mlc
        chmod +x mlc/Linux/mlc
    fi
    uv run python -c "import config; config.tune_machine()"
    bw() { $@ mlc/Linux/mlc --loaded_latency -d0 -t20 -R -b200m | tail -1 | awk '{print $3}'; }
    local=$(bw)
    il=$(bw numactl --interleave=all)
    echo "local $local MB/s, interleaved $il MB/s"
    awk -v l=$local -v i=$il 'BEGIN { printf "read_penalty %.0f\n", 100 * (l / i - 1) }'

bench-fio:
    uv run run.py fio
    uv run run.py fio-written-default

bench-fio-repl:
    uv run run.py fio-repl
    uv run run.py fio-written

bench-fio-pgt-spare:
    uv run run.py fio-pgt-spare

bench-fio-pgt-mitosis:
    uv run run.py fio-pgt-mitosis

bench-fio-pgt-hydra:
    uv run run.py fio-pgt-hydra

bench-llama:
    #!/usr/bin/env bash
    . /opt/intel/oneapi/setvars.sh
    uv run run.py llama

bench-duckdb:
    uv run run.py duckdb

bench-duckdb-repl:
    uv run run.py duckdb-repl

bench-llama-repl:
    #!/usr/bin/env bash
    . /opt/intel/oneapi/setvars.sh
    uv run run.py llama-repl

# rebuild the llama csvs from the per-run jsonl, no re-run
llama-csv:
    uv run run.py llama-csv

# cross socket line sharing; needs no repl kernel, measures what repl removes
bench-sharing: build-dirtest
    uv run run.py sharing

plot-ann:
    uv run run.py plot-ann

plot-rocksdb:
    uv run run.py plot-rocksdb

plot-llama:
    uv run run.py plot-llama

plot-fio:
    uv run run.py plot-fio

plot-fio-pgtable:
    uv run run.py plot-fio-pgtable

bench-pgtable-sync-baseline:
    uv run run.py bench-pgtable-sync-baseline

bench-pgtable-sync-repl:
    uv run run.py bench-pgtable-sync-repl

plot-pgtable-sync:
    uv run run.py plot-pgtable-sync

plot-microbench:
    uv run run.py plot-microbench

plot-pressure:
    uv run run.py plot-pressure

plot-monitoring:
    uv run run.py plot-monitoring

plot-sharing:
    uv run run.py plot-sharing

plot-duckdb:
    uv run run.py plot-duckdb

stats:
    uv run run.py stats-monitoring

stats-pressure:
    uv run run.py stats-pressure

analyze-runtime:
    uv run python analyze_runtime.py

# what the paper uses, laid out clean in another repo: just paper ~/paper/data
# (the destination is wiped first; anything listed but not generated is reported)
paper dest:
    #!/usr/bin/env bash
    set -euo pipefail
    D="{{dest}}"
    rm -rf "$D"; mkdir -p "$D"
    put() { # put <dest dir> <source>...
        local to="$D/$1"; shift; mkdir -p "$to"
        for f in "$@"; do
            if [ -e "$f" ]; then cp -r "$f" "$to/"; else echo "missing: $f"; fi
        done
    }
    P=plots
    put figures/ann $P/ann/gold_ann.pdf $P/ann/silver_ann.pdf $P/ann/legend.pdf
    put figures/rocksdb $P/rocksdb/gold_rocksdb.pdf $P/rocksdb/silver_rocksdb.pdf
    put figures/llama $P/llama/gold_llama.pdf $P/llama/silver_llama.pdf
    put figures/pgtable_sync $P/microbench/pgtable_sync/{gold,silver,silver_gold,legend}.pdf
    put figures/fio $P/fio/{gold,silver}_{uniform,written}_{read,write}.pdf $P/fio/legend.pdf \
        $P/fio/{gold,silver}_fio_pgtable.pdf
    put figures/pressure $P/pressure/gold_qps.pdf $P/pressure/silver_qps.pdf $P/pressure/qps_legend.pdf
    put figures-details $P/ann/{gold,silver}_ann_abs.pdf $P/rocksdb/{gold,silver}_rocksdb_abs.pdf \
        $P/llama/{gold,silver}_llama_abs.pdf $P/pressure/details/*
    for m in "gold-x4 IntelR_XeonR_Gold_6130_CPU_@_2.10GHz_X86_64" \
             "silver-x2 IntelR_XeonR_Silver_4216_CPU_@_2.10GHz_X86_64"; do
        set -- $m; n=$1; R="results/$2"
        put stats/$n $R/stats/*.csv $R/stats/compare $R/stats/details
        put results/$n/ann $R/ann/*.csv
        put results/$n/fio $R/fio/*.csv $R/fio/*.jsonl
        put results/$n/llama $R/llama/*
        put results/$n/locality $R/locality/*.csv
        put results/$n/pgtable_sync $R/microbench/pgtable_sync/*
        put results/$n/pressure $R/pressure/*
        put results/$n/rocksdb $R/rocksdb/results.csv
        # per run: db_bench's own summary and result csv, not the raw logs
        for r in $R/rocksdb/outputs/*/; do
            [[ $r == *_load/ ]] && continue
            put results/$n/rocksdb/outputs/$(basename $r) $r/report.tsv $r/*.r.csv
        done
    done
    du -sh "$D"

build: build-rocksdb build-fio build-micro

bench: bench-ann bench-rocksdb
bench-repl: bench-ann-repl bench-rocksdb-repl

plot: plot-ann plot-rocksdb

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

bench-rocksdb:
    uv run run.py rocksdb

bench-rocksdb-repl:
    uv run run.py rocksdb-repl

bench-ann:
    uv run run.py ann

bench-ann-repl:
    uv run run.py ann-repl

plot-ann:
    uv run run.py plot-ann

plot-rocksdb:
    uv run run.py plot-rocksdb

plot-monitoring:
    uv run run.py plot-monitoring

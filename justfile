build: build-rocksdb

bench: bench-ann bench-rocksdb
bench-repl: bench-ann-repl bench-rocksdb-repl

build-rocksdb:
    #!/usr/bin/env bash
    set -euxo pipefail
    mkdir -p rocksdb/build
    cd rocksdb/build
    cmake -DCMAKE_BUILD_TYPE=Release -DFAIL_ON_WARNINGS=OFF -DWITH_ZSTD=ON ..
    make -j

bench-rocksdb:
    uv run run.py rocksdb

bench-rocksdb-repl:
    uv run run.py rocksdb-repl

bench-ann:
    uv run run.py ann

bench-ann-repl:
    uv run run.py ann-repl



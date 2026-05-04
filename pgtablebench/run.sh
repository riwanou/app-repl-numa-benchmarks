#!/bin/bash

# spare | mitosis | hydra
SYSTEM=${SYSTEM:-spare}
echo "System $SYSTEM"

REPL_POLICY=/sys/kernel/debug/repl_pt/policy
MITOSIS_NUMACTL=/root/linux-mitosis-4.17/mitosis-numactl/numactl
HYDRA_NUMACTL=/root/linux-hydra-6.5/hydra-numactl/numactl

FLAMEGRAPH_DIR=./FlameGraph
PERF_FREQ=999

run_repl() {
    local cmd="$@"
    case $SYSTEM in
        spare)
            echo 1 > $REPL_POLICY
            $cmd
            echo 0 > $REPL_POLICY
            ;;
        mitosis)
            echo -1 > /proc/sys/kernel/pgtable_replication_cache
            $MITOSIS_NUMACTL --pgtablerepl=all -- $cmd
            ;;
        hydra)
            echo 1 > /proc/sys/vm/hydra_repl_order
            echo 1 > /proc/sys/vm/hydra_tlbflush_opt
            $HYDRA_NUMACTL --pgtablerepl=all -- $cmd
            ;;
    esac
}

flame() {
    local label=$1
    shift 1
    local cmd="$@"

    perf record -F 999 -g -e cycles:k -o perf_data/perf_${label}.data -- $cmd
    perf script -i perf_data/perf_${label}.data | \
        $FLAMEGRAPH_DIR/stackcollapse-perf.pl | \
        $FLAMEGRAPH_DIR/flamegraph.pl --title "$label" > perf_data/flame_${label}.svg
    echo "Flamegraph: /tmp/flame_${label}.svg"

    perf probe --add 'repl_prepare_zap_pte_folio'
    perf probe --add 'repl_should_not_zap_page'

    # echo "=== TLB stats for $label ==="
    # perf stat -e tlb:tlb_flush \
    #           -- "$@" 2>&1
    perf stat -e tlb:tlb_flush \
              -e probe:repl_prepare_zap_pte_folio \
              -e probe:repl_should_not_zap_page \
              -- "$@" 2>&1

    perf probe --del repl_prepare_zap_pte_folio
    perf probe --del repl_should_not_zap_page
}

run_flamegraphs() {
    local npages=$1
    local nrounds=$2

    echo "=== Flamegraph Non Replicated ==="
    # warmup
    ./bench -s $npages -n 64 > /dev/null
    flame "norepl_${npages}" \
        ./bench -s $npages -n $nrounds

    echo "=== Flamegraph Replicated ==="
    # warmup
    run_repl ./bench -s $npages -n 64 > /dev/null
    case $SYSTEM in
        spare)
            echo 1 > $REPL_POLICY
            flame "repl_${npages}" \
                ./bench -s $npages -n $nrounds
            echo "=== Flamegraph Replicated Data ==="
            flame "repl_${npages}_data" \
                ./bench -s $npages -n $nrounds -r
            echo 0 > $REPL_POLICY
            ;;
        mitosis)
            flame "repl_${npages}" "/tmp/flame_repl.svg" \
                $MITOSIS_NUMACTL --pgtablerepl=all -- ./bench -s $npages -n $nrounds
            ;;
        hydra)
            flame "repl_${npages}pages" "/tmp/flame_repl.svg" \
                $HYDRA_NUMACTL --pgtablerepl=all -- ./bench -s $npages -n $nrounds
            ;;
    esac
}

run_bench() {
    local npages=$1
    local nrounds=$2
    echo "NPAGES=$npages"

    echo "===Non Replicated==="
    ./bench -s $npages -n $nrounds > /dev/null
    ./bench -s $npages -n $nrounds

    echo "===Replicated Page tables only==="
    ./bench -s $npages -n $nrounds > /dev/null
    run_repl ./bench -s $npages -n $nrounds

    if [ $SYSTEM == "spare" ]; then
      echo "===Replicated Page tables and Data==="
      run_repl ./bench -s $npages -n $nrounds -r
    fi
}

mkdir -p perf_data
gcc -Wall bench.c -o bench -lnuma -lpthread -lm
echo 3 > /proc/sys/vm/drop_caches

run_bench 32768 100

# run_flamegraphs 32768 10
# run_flamegraphs 16384 50

# echo 1 > $REPL_POLICY
# flame "repl_2048_128_simple_repl" \
#     ./bench -s 2048 -n 128 -r
# echo 0 > $REPL_POLICY

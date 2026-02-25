#!/bin/bash

session_name="$1"
workdir="$2"
shift 2
cmd="$*"

# perf governor
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# disable smt
# echo off > /sys/devices/system/cpu/smt/control
# disable frequency boosting
# echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo


if [ ! -d "$workdir/app-repl-numa-benchmarks" ]; then
  git clone https://github.com/riwanou/app-repl-numa-benchmarks.git "$workdir/app-repl-numa-benchmarks"
fi

if [ ! -f "$workdir/app-repl-numa-benchmarks/rocksdb/README.md" ]; then
  (cd "$workdir/app-repl-numa-benchmarks" && git submodule update --init --recursive)
fi

existing_session=$(tmux list-sessions 2>/dev/null | grep '^bench-' || true)

if [[ -n "$existing_session" ]]; then
  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Attaching to existing tmux session: $session_name"
    tmux set-option -t "$session_name" mouse on
    exec tmux attach-session -t "$session_name"
  else
    echo "Error: A different bench tmux session already exists:"
    echo "$existing_session"
    echo "Please close that session first before starting a new one."
    exit 1
  fi
else
  echo "Creating new tmux session: $session_name"
  tmux new-session -d -s "$session_name" "$cmd"
  tmux set-option -t "$session_name" mouse on
  exec tmux attach-session -t "$session_name"
fi


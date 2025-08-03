#!/bin/bash

workdir="$1"
virtu_config="$2"
meta_data="$3"
image_path="$4"
seed_name="$5"

user_data="virtu/user-data"
session_name="virtu"

if tmux has-session -t "$session_name" 2>/dev/null; then
   echo "Attaching to existing tmux session: $session_name"
   tmux set-option -t "$session_name" mouse on
   exec tmux attach-session -t "$session_name"
else
  cd $workdir
  genisoimage -output $seed_name -volid cidata -joliet -rock $user_data $meta_data
  chmod +x $virtu_config

  echo "Creating new tmux session: $session_name"
  echo "Virtu config: $virtu_config"
  tmux new-session -d -s "$session_name" "IMAGE=${image_path} SEED=${seed_name} ./${virtu_config}"
  tmux set-option -t "$session_name" mouse on
  exec tmux attach-session -t "$session_name"
fi

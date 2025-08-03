#!/bin/bash
set -e

echo "Installing dependencies"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl build-essential pcm libomp-dev libnuma-dev numactl time sysstat \
  cmake flex bison bc libncurses-dev libelf-dev libssl-dev kexec-tools \
  tmux libgflags-dev clang libzstd-dev psmisc
apt-get install -y ./helix_25.1.1-1_amd64.deb ./zenith_0.14.1-1_amd64.deb

echo "Installing tools"
curl --proto '=https' --tlsv1.2 -LsSf https://setup.atuin.sh | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
cargo install just

# Ensure atuin is initialized only once
LINE='eval "$(atuin init bash --disable-up-arrow)"'
sed -i '/atuin init bash/d' ~/.bashrc
grep -qxF "$LINE" ~/.bashrc || echo "$LINE" >> ~/.bashrc
echo "✅ Finished setting up dependencies and tools."

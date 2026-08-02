#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

echo "Installing dependencies"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl build-essential pcm libomp-dev libnuma-dev numactl time sysstat \
  cmake flex bison bc libncurses-dev libelf-dev libssl-dev kexec-tools \
  tmux libgflags-dev clang libzstd-dev psmisc rsync sudo linux-perf
apt-get install -y ./helix_25.1.1-1_amd64.deb ./zenith_0.14.1-1_amd64.deb

echo "Installing tools"
curl --proto '=https' --tlsv1.2 -LsSf https://setup.atuin.sh | sh < /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
cargo install just

# Ensure atuin is initialized only once
LINE='eval "$(atuin init bash --disable-up-arrow)"'
sed -i '/atuin init bash/d' ~/.bashrc
grep -qxF "$LINE" ~/.bashrc || echo "$LINE" >> ~/.bashrc
echo "✅ Finished setting up dependencies and tools."

# Docker
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: $(. /etc/os-release && echo "$VERSION_CODENAME")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

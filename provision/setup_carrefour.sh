#!/bin/bash
set -e


echo "Setup old apt sources"
tee /etc/apt/sources.list <<EOF
deb http://archive.ubuntu.com/ubuntu trusty main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu trusty-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu trusty-security main restricted universe multiverse
EOF

echo "Installing Carrefour dependencies"
echo kexec-tools kexec-tools/load_kexec boolean false | debconf-set-selections
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --force-yes \
    git curl build-essential libnuma-dev numactl \
    time sysstat cmake flex bison bc libncurses-dev kexec-tools \
    tmux libgflags-dev clang psmisc

if [ ! -d "linux-carrefour" ]; then
    git clone https://github.com/Carrefour/linux-replication.git \
    "linux-carrefour"
fi

echo "Building Carrefour's linux replication kernel"
cd linux-carrefour
cp config-bench .config
make -j$(nproc)

echo "Install Carrefour's linux replication kernel"
make modules_install -j$(nproc)
make install

echo "Restarting using Carrefour's linux replication kernel"
update-grub
grub-reboot "gnulinux-advanced-29b74d93-f08b-42be-ae5c-a61723438b4e>gnulinux-3.6.0-replication+-advanced-29b74d93-f08b-42be-ae5c-a61723438b4e"
sync
reboot

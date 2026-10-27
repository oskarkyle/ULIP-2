#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ULIP-2
mkdir -p cache/apt/lists/partial cache/apt/archives/partial cache/apt/debs env/sysroot
opts=(-o Dir::State::lists=/workspace/ULIP-2/cache/apt/lists -o Dir::Cache::archives=/workspace/ULIP-2/cache/apt/archives -o Debug::NoLocking=1 -o APT::Get::List-Cleanup=0)
apt-get "${opts[@]}" update
cd cache/apt/debs
apt-get "${opts[@]}" download libx11-6 libxcb1 libxau6 libxdmcp6 libbsd0 libgl1 libglx0 libglvnd0 libc++1 libc++1-10 libc++abi1-10 libunwind8
for package in *.deb; do dpkg-deb -x "$package" /workspace/ULIP-2/env/sysroot; done

#!/usr/bin/env bash
set -euo pipefail
base=/media/robosense/DataCenterNew/Kyle/ULIP-2
for step in install_system_libs install_env build_and_smoke; do
  bash "$base/setup/container.sh" "/workspace/ULIP-2/setup/$step.sh" 2>&1 | tee "$base/outputs/$step.log"
done

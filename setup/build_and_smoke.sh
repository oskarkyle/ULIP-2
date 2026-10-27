#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ULIP-2
source env/ulip/bin/activate
python -m pip install --no-build-isolation --no-deps ./setup/vendor/pointnet2/pointnet2_ops_lib
cd repo
python -u ../setup/smoke_pointbert.py
python -m pip freeze > ../outputs/python-freeze.txt

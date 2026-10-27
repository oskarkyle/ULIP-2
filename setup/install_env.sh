#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ULIP-2
mkdir -p cache/pip cache/tmp cache/torch_extensions cache/huggingface cache/torch cache/home
python -m venv --system-site-packages env/ulip
source env/ulip/bin/activate
python -m pip install 'pip<25.1' 'setuptools<70' wheel
python -m pip install 'numpy==1.23.5' 'timm==0.9.16' 'open-clip-torch==2.24.0' 'huggingface-hub==0.23.5' 'easydict==1.9' 'ftfy==6.0.1' 'regex==2022.10.31' 'pyyaml_env_tag==0.1' 'h5py==3.6.0' 'open3d==0.16.0' 'wandb==0.13.3' 'termcolor==1.1.0' 'lmdb==1.3.0' 'protobuf==3.19.6' 'dash==2.18.2' 'plotly==5.24.1' 'pydantic==1.8.2' 'Pillow==9.5.0' 'sentencepiece==0.1.99' 'ninja' 'setuptools<70'
python -m pip freeze > outputs/python-freeze.txt
python -c 'import torch,open_clip,timm,open3d; print("INSTALLED",torch.__version__,timm.__version__,open3d.__version__); print("CUDA",torch.cuda.is_available())'

#!/bin/bash
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "$WORK_DIR/../../../" && pwd)"
echo "TOP_DIR is ${TOP_DIR}"

#install nginx
cd ${TOP_DIR}/omni/accelerators/sched/omni_proxy/
bash build.sh --skip-extras -c
#install vllm
cd ${TOP_DIR}/infer_engines/
sh bash_install_code.sh
cd ${TOP_DIR}/infer_engines/vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e .
cd ${TOP_DIR}
pip3 install gcovr
pip3 install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip3 install torch-npu==2.6.0
#install omniinfer
pip install -e .

#!/bin/bash
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "$WORK_DIR/../../../" && pwd)"
echo "TOP_DIR is ${TOP_DIR}"

#download model json file
cd ${WORK_DIR}
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json -P ./mock_model/
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer.json -P ./mock_model/
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json -P ./mock_model/
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/vocab.json -P ./mock_model/

#install nginx
pkill nginx
cd ${TOP_DIR}/omni/accelerators/sched/omni_proxy/
bash build.sh --skip-extras -c
pip3 install gcovr

#install vllm
cd ${TOP_DIR}/infer_engines/
sh bash_install_code.sh
cd ${TOP_DIR}/infer_engines/vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e .

#install omniinfer
cd ${TOP_DIR}
pip3 install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip3 install torch-npu==2.6.0
pip install -e .

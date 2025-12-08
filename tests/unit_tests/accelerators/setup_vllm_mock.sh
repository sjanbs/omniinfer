cd ../../../infer_engines/
sh bash_install_code.sh
cd vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e .
cd ../..
pip3 install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip3 install torch-npu==2.6.0
pip install -e .

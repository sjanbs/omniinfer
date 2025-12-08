import pytest
import os
import subprocess
import time
import requests
import json
from pathlib import Path

# Configuration
LOG_FILE_PREFIX = "server"
APP_START_MARKER = "Application startup complete."
STARTUP_TIMEOUT = 30  # seconds
PREFILL_NUM = 3
PREFILL_BASE_API_PORT=8000
DECODE_NUM = 3
DECODE_BASE_API_PORT=9000
PROXY_PORT=7000
model_path="/mnt/sfs_turbo/bucket-910c-6055/models/Qwen2.5-7B-Instruct/"
tp=1
dp=1

processes = []
log_files = []
proxy_script_path = "../../../omni/accelerators/sched/omni_proxy/omni_proxy.sh"

def cleanup_subprocess():
    for i, proc in enumerate(processes):
        if proc.poll() is None:
            print(f"[TEARDOWN] Terminating instance {i} pid {proc.pid}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"[TEARDOWN] Killing instance {i} pid {proc.pid}...")
                proc.kill()
                proc.wait()
        else:
            print(f"[TEARDOWN] Instance {i} pid {proc.pid} already exited (code: {proc.returncode}).")
    # Optional: auto-cleanup logs
    # for lf in log_files:
    #     if lf.exists():
    #         lf.unlink()

def generate_proxy_endpoints(port: int, num: int) -> str:
    return ",".join(f"127.0.0.1:{port + i}" for i in range(num))

def setup_proxy():
    prefill_list = generate_proxy_endpoints(PREFILL_BASE_API_PORT, PREFILL_NUM)
    decode_list = generate_proxy_endpoints(DECODE_BASE_API_PORT, DECODE_NUM)
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", "/usr/local/nginx/conf/nginx.conf",
            "--core-num", "1",
            "--listen-port", f"{PROXY_PORT}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", "/usr/local/nginx/logs/error.log",
            "--log-level", "info",
            "--access-log-file", "/usr/local/nginx/logs/access.log",
        ]
        print(f"\n[SETUP] Starting proxy with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        pytest.fail(error_msg)

def setup_vllm(is_prefill, node_num):
    env = os.environ.copy()
    env['VLLM_ENABLE_MC2'] = '0'
    env['VLLM_USE_V1'] = '1'
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
    env['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = "1"
    env['HCCL_CONNECT_TIMEOUT'] = "3600"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    env["USING_LCCL_COM"] = "0"
    env["NO_NPU_MOCK"] = "1"
    env["RANDOM_MODE"] = "1"
    env["KV_CACHE_MODE"] = "1"
    start_port = DECODE_BASE_API_PORT
    node_type = "decode"
    if is_prefill:
        start_port = PREFILL_BASE_API_PORT
        node_type = "prefill"
        os.environ["PREFILL_PROCESS"] = "1"

    for idx in range(node_num):
        cmd = [
            "vllm", "serve", model_path,
            "--port", f"{start_port + idx}",
            "--max_num_seqs", "128",
            "--max_model_len", "8000",
            "--tensor_parallel_size", f"{tp}",
            "--data_parallel_size", f"{dp}",
            "--gpu_memory_utilization", "0.9",
            "--trust_remote_code",
            "--served-model-name", "deepseek",
            "--dtype", "bfloat16",
            "--distributed-executor-backend", "mp",
            "--block_size", "128",
        ]
        log_file = Path(f"{LOG_FILE_PREFIX}_{idx}.log")
        log_files.append(log_file)

        # Clean existing log
        if log_file.exists():
            log_file.unlink()
        if idx == 0:
            print(f"\n[SETUP] Starting {node_num} {node_type} background processes with command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print(f"[SETUP] Starting {node_type} instance {idx}, pid {proc.pid} -> {log_file}")
        processes.append(proc)

@pytest.fixture(scope="module")
def setup_teardown():
    setup_proxy()
    setup_vllm(True, PREFILL_NUM)
    setup_vllm(False, DECODE_NUM)

    total_node_num = PREFILL_NUM + DECODE_NUM
    # --- Wait for all instances to emit "APP started" ---
    start_time = time.time()
    ready_status = [False] * total_node_num

    while time.time() - start_time < STARTUP_TIMEOUT:
        all_ready = True
        for i, (proc, log_file) in enumerate(zip(processes, log_files)):
            if ready_status[i]:
                continue

            # Check if process crashed
            if proc.poll() is not None:
                cleanup_subprocess()
                pytest.fail(f"Instance {i} pid {proc.pid} exited early with code {proc.returncode}. Check {log_file}")

            # Check log for start marker
            if log_file.exists():
                with open(log_file, "r") as f:
                    if APP_START_MARKER in f.read():
                        ready_status[i] = True
                        print(f"[SETUP] Instance {i} pid {proc.pid} started successfully.")

            all_ready = all_ready and ready_status[i]

        if all_ready:
            break

        time.sleep(1)

    # --- Handle timeout ---
    if not all(ready_status):
        failed = [i for i, r in enumerate(ready_status) if not r]
        cleanup_subprocess()
        pytest.fail(
            f"{len(failed)} instance(s) did not start within {STARTUP_TIMEOUT}s: {failed}. "
            f"Check logs: {[str(log_files[i]) for i in failed]}"
        )

    print(f"[SETUP] All {total_node_num} instances are ready.")

    yield {"processes": processes, "log_files": log_files}

    # --- Teardown: Shut down all instances ---
    print(f"\n[TEARDOWN] Shutting down {total_node_num} instances...")
    cleanup_subprocess()


def test_chat_completions(setup_teardown):
    url = f"http://127.0.0.1:{PREFILL_BASE_API_PORT}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

def test_chat_completions_stream(setup_teardown):
    url = f"http://127.0.0.1:{PREFILL_BASE_API_PORT}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }
    # Enable streaming response handling
    with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as resp:
        resp.raise_for_status()  # Raise exception for HTTP error status codes
        token_cnt = 0
        for line in resp.iter_lines():
            if line:
                # Process Server-Sent Events (SSE) lines
                if line.startswith(b"data:"):
                    json_str = line[len(b"data:"):].strip()
                    if json_str == b"[DONE]":
                        print(f"Stream finished. get {token_cnt} output\n")
                        break
                    try:
                        chunk = json.loads(json_str)
                        # Extract content (assuming OpenAI-compatible format)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            token_cnt += 1
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"\nFailed to decode JSON: {json_str}")

def test_chat_completions_invalid_server(setup_teardown):
    url = f"http://127.0.0.1:{PREFILL_BASE_API_PORT-1}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    with pytest.raises(requests.exceptions.ConnectionError):
        response = requests.post(url, headers=headers, json=data, timeout=3)

def test_chat_completions_with_proxy(setup_teardown):
    url = f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

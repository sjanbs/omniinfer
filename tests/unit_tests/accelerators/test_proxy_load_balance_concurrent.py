import pytest
import os
import subprocess
import time
import json
from pathlib import Path
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import strart_vllm_mock, cleanup_subprocess
import port_manager
import requests
import concurrent.futures
import random
import re
from collections import defaultdict
from pathlib import Path

# Configuration
PREFILL_NUM = 4
DECODE_NUM = 4
proxy_port = 7000
prefill_port_list = None
decode_port_list = None
CUR_DIR = Path(__file__).parent

@pytest.fixture(scope="module")
def setup_teardown():
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        return

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    processes = strart_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start vllm fail")

    yield

    # --- Teardown: Shut down all instances ---
    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)
def fetch_post(url, headers, data):
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)  
        return {
            "url": url,
            "status": response.status_code,
            "text": response.text[:200] + "..." if len(response.text) > 200 else response.text
        }
    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "error": str(e)
        }

def test_chat_completions_with_proxy(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"

    json_path = "300024096.json"  
    with open(json_path, 'r') as f:
        stories = json.load(f)  

    start_time = time.time()
    results = []
    POST_DATA_COUNT = 2000

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(PREFILL_NUM, 200)) as executor:
        futures = []
        
        for _ in range(POST_DATA_COUNT):
            content_input = random.choice(stories)
            content = content_input["input"]

            data = {
                "model": "deepseek",
                "temperature": 0,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": content}],
                "stream": True
            }

            headers = {
                "Content-Type": "application/json",
                "X-Request-Id": ''.join(random.choices('123456789', k=5)),
            }

            future = executor.submit(fetch_post, url, headers, data)
            futures.append(future)
            # time.sleep(0.05)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                # print(f"Status: {result['status']}, Preview: {result.get('text', 'Error')}")
                assert result.get('status') == 200
            except Exception as e:
                print(f"Error: {e}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    log_file = f"{CUR_DIR}/nginx_access.log"
    num_logs = POST_DATA_COUNT
    print("\n=== verifying load balance ===")
    try:
        analysis_result = analyze_balance(log_file, num_logs)
        for idx in sorted(analysis_result['prefill_frequency'].keys()):
            count = analysis_result['prefill_frequency'][idx]
            assert num_logs // 4 - 20 <= count <= num_logs // 4 + 20

        for idx in sorted(analysis_result['decode_frequency'].keys()):
            count = analysis_result['decode_frequency'][idx]
            assert num_logs // 4 - 20 <= count <= num_logs // 4 + 20

    except Exception as e:
        print(f"\n=== verifying fail: {e} ===")
        raise
    print("\n=== verifying pass ===")

def analyze_balance(log_file, num_logs):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"log {log_file} does not exist")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"failed to read log: {e}")
    
    if not logs:
        raise ValueError("empty log")
    
    prefill_freq = defaultdict(int)
    decode_freq = defaultdict(int)
    
    recent_logs = logs[-num_logs:] 
    
    for line in recent_logs:
        try:
            data = parse_log_line(line)
            if not data:
                continue
                
            prefill_val = data.get('prefill_idx')
            decode_val = data.get('decode_idx')
            
            if prefill_val is not None:
                prefill_freq[int(prefill_val)] += 1
            if decode_val is not None:
                decode_freq[int(decode_val)] += 1
        except Exception as e:
            print(f"Error parsing line: {e} (log: {line[:100]}...)")
            continue

    return {
        'prefill_frequency': dict(prefill_freq),
        'decode_frequency': dict(decode_freq)
    }

def parse_log_line(line):
    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None  
    line = line[1:-1]  
    parts = []
    current = ""
    in_string = False 
    for char in line:
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif char == "," and not in_string:
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    result = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            result[key] = value[1:-1]
        elif value.replace(".", "", 1).isdigit():
            result[key] = float(value) if "." in value else int(value)
        else:
            result[key] = value
    
    return result

import pytest
from unittest.mock import MagicMock, patch
import os
import requests

from infer_engines.vllm.vllm.entrypoints.openai import serving_engine 
from infer_engines.vllm.vllm.engine.multiprocessing.client import MQLLMEngineClient

from test_proxy import setup_teardown


def test_tokenizer_worker_num_cannot_exceed_available_cores():
    env_vars = {
        "TOKENIZER_PROC_POOL": "1",
        "TOKENIZER_WORKER_NUM": "6",
        "TOKENIZER_PROC_POOL_THRES": "128",
        "TOKENIZER_AFFINITY_CORES": "0,1,2,3"
    }

    # Use patch.dict to temporarily overwrite os.environ
    with patch.dict(os.environ, env_vars):
        mock_engine_client = MagicMock()
        mock_model_config = MagicMock()
        mock_model_config.max_model_len = 1024
        mock_models = MagicMock()
        mock_request_logger = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            openai_serving = serving_engine.OpenAIServing(
                engine_client=mock_engine_client,
                model_config=mock_model_config,
                models=mock_models,
                request_logger=mock_request_logger,
                return_tokens_as_token_ids=False,
            )

    assert str(exc_info.value) == "tokenizer_worker_num (6) cannot exceed available_cores (4)"

def test_api_server_health(setup_teardown):
    from test_proxy import prefill_port_list

    url = f"http://127.0.0.1:{prefill_port_list[0]}/health"
    response = requests.get(url, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

    try:
        file_path = "/opt/cloud/node/npu_status.yaml"
        path_exists = os.path.exists(file_path)
        if not path_exists:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                mock_yaml_content = """
        resources:
        - status: 
            - npu: 0
              errLevel: L1
        - errLevelName: NotHandle
        """
                f.write(mock_yaml_content)

        url = f"http://127.0.0.1:{prefill_port_list[0]}/health"
        response = requests.get(url, timeout=10)
        print("Status Code:", response.status_code)
        print("Response Headers:", response.headers)
        print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        assert response.status_code == 200
    finally:
        if not path_exists:
            os.remove(file_path)

def test_completion_using_process_pool(setup_teardown):
    from test_proxy import prefill_port_list

    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }

    # Contruct a long enough prompt that will enable process pool
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "prompt": list("""Act as an expert creative writing instructor and narrative analyst. 
                    Generate a detailed, original fantasy world setting in 3-4 paragraphs. 
                    Include unique geographical features, a brief historical conflict, 
                    and a description of at least two distinct cultures or societies within it."""),
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

def test_chat_completion_using_proc_pool(setup_teardown):
    from test_proxy import prefill_port_list

    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    
    # Contruct a long enough prompt that will enable process pool
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", 
                        "prefix": True,
                        "content": """Generate a detailed, original fantasy world setting in 3-4 paragraphs. 
                                    Include unique geographical features, a brief historical conflict, and 
                                    a description of at least two distinct cultures or societies within it."""}],
        "stream": True,
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

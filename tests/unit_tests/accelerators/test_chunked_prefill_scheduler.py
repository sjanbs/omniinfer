import os
from pathlib import Path
from unittest.mock import patch
import sys

def test_no_chunk():
    """Prefills should not be chunked without `FORCE_ENABLE_CHUNK_PREFILL` set."""
    
    from infer_engines.vllm.tests.v1.core.test_scheduler import create_scheduler, create_requests

    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64

    # Use cached model 
    CUR_DIR = Path(__file__).parent
    model_path=f"{CUR_DIR}/mock_model/"

    scheduler = create_scheduler(model=model_path,
                                max_num_seqs=max_seqs,
                                max_num_batched_tokens=max_num_batched_tokens,
                                num_blocks=32,
                                block_size= block_size,
                                max_model_len=max_model_len,)
    requests = create_requests(num_requests=2,
                            num_tokens=60)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1
    assert len(output.scheduled_new_reqs) == 1
    assert len(output.scheduled_cached_reqs) == 0
    assert output.num_scheduled_tokens[requests[1].request_id] == 60
    assert output.total_num_scheduled_tokens == 60
    requests[1].append_output_token_ids(1)

    output = scheduler.schedule()
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    assert len(output.scheduled_new_reqs) == 1
    assert len(output.scheduled_cached_reqs) == 1
    assert output.num_scheduled_tokens[requests[0].request_id] == 60
    assert output.num_scheduled_tokens[requests[1].request_id] == 1
    assert output.total_num_scheduled_tokens == 61

def test_chunk():
    """Verify prefills are chunked properly."""

    env_vars = {
        "FORCE_ENABLE_CHUNK_PREFILL": "1",
    }

    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64

    # Use cached model 
    CUR_DIR = Path(__file__).parent
    model_path=f"{CUR_DIR}/mock_model/"

    with patch.dict(os.environ, env_vars):
        # Clear module cache 
        if 'infer_engines.vllm.tests.v1.core.test_scheduler' in sys.modules:
            del sys.modules['infer_engines.vllm.tests.v1.core.test_scheduler']
        if 'vllm.v1.core.sched.scheduler' in sys.modules:
            del sys.modules['vllm.v1.core.sched.scheduler']

        # Reload module with new env var
        from infer_engines.vllm.tests.v1.core.test_scheduler import create_scheduler, create_requests

        scheduler = create_scheduler(model=model_path,
                                    max_num_seqs=max_seqs,
                                    max_num_batched_tokens=max_num_batched_tokens,
                                    num_blocks=32,
                                    block_size= block_size,
                                    max_model_len=max_model_len,)
        requests = create_requests(num_requests=2,
                                num_tokens=60)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()
        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == len(requests)
        assert len(output.scheduled_new_reqs) == len(requests)
        assert len(output.scheduled_cached_reqs) == 0
        assert output.num_scheduled_tokens[requests[0].request_id] == 4
        assert output.num_scheduled_tokens[requests[1].request_id] == 60
        assert output.total_num_scheduled_tokens == 64
        requests[1].append_output_token_ids(1)

        output = scheduler.schedule()
        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == len(requests)
        assert len(output.scheduled_new_reqs) == 0
        assert len(output.scheduled_cached_reqs) == len(requests)
        assert output.num_scheduled_tokens[requests[0].request_id] == 56
        assert output.num_scheduled_tokens[requests[1].request_id] == 1
        assert output.total_num_scheduled_tokens == 57

    if 'infer_engines.vllm.tests.v1.core.test_scheduler' in sys.modules:
        del sys.modules['infer_engines.vllm.tests.v1.core.test_scheduler']
    if 'vllm.v1.core.sched.scheduler' in sys.modules:
        del sys.modules['vllm.v1.core.sched.scheduler']

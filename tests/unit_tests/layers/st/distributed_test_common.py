import os
import pytest
import tempfile
import traceback
import importlib
from typing import Callable, Any, List, Tuple
import torch.multiprocessing as mp
import torch
TEST_SEED = 0

def _persistent_worker_loop(rank: int, world_size: int, temp_file_path: str, 
                            task_queue: mp.Queue, result_queue: mp.Queue):
    try:
        # 1. Apply Patches Immediately
        from omni.adaptors.vllm.patches.model_patch import patch_all
        patch_all()

        # 2. Set Configuration BEFORE loading/reloading layers
        #    This is the fix: Ensure config is correct before the layer class binds to it.
        from omni.models.config_loader.loader import model_extra_config
        model_extra_config.parall_config.dense_mlp_tp_size = world_size
        model_extra_config.parall_config.o_proj_tp_size = world_size

        # 3. CRITICAL: Reload the layer module
        #    This forces the layer classes to re-read the configuration we just set.
        import omni.layers.linear
        importlib.reload(omni.layers.linear)

        # 4. Setup Distributed Environment
        from vllm import distributed as vllm_dist
        from vllm.utils import update_environment_variables
        
        torch.npu.set_device(rank)
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        update_environment_variables({
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        })

        vllm_dist.init_distributed_environment(
            distributed_init_method=f"file://{temp_file_path}",
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            backend="hccl",
        )
        
        vllm_dist.initialize_model_parallel(tensor_model_parallel_size=world_size)
        
        # Verify TP Size
        current_tp = vllm_dist.parallel_state.get_tensor_model_parallel_world_size()
        if current_tp != world_size:
            raise RuntimeError(f"Distributed Init Failed: Expected TP={world_size}, got {current_tp}")

        # 5. Signal Ready
        result_queue.put("READY")

        # 6. Task Loop
        while True:
            task = task_queue.get()
            if task is None: break
            
            func, args, kwargs = task
            
            try:
                torch.manual_seed(TEST_SEED)
                func(rank, world_size, *args, **kwargs)
                result_queue.put(None) 
            except Exception:
                tb = traceback.format_exc()
                result_queue.put(RuntimeError(f"Rank {rank} failed:\n{tb}"))
                
    except Exception:
        tb = traceback.format_exc()
        result_queue.put(RuntimeError(f"Worker Startup Failed Rank {rank}:\n{tb}"))
    finally:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except:
            pass

@pytest.fixture(scope="module")
def distributed_worker_pool():
    world_size = 2
    ctx = mp.get_context('spawn') # Use spawn context
    
    task_queues = [ctx.Queue() for _ in range(world_size)]
    result_queues = [ctx.Queue() for _ in range(world_size)]
    
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        temp_file_path = tfile.name

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_persistent_worker_loop,
            args=(rank, world_size, temp_file_path, task_queues[rank], result_queues[rank])
        )
        p.start()
        processes.append(p)

    try:
        for i, q in enumerate(result_queues):
            res = q.get(timeout=30)
            if res != "READY":
                raise RuntimeError(f"Worker {i} failed to start: {res}")
    except Exception as e:
        for p in processes: p.terminate()
        raise e

    def run_task(func, *args, **kwargs):
        for q in task_queues:
            q.put((func, args, kwargs))
        
        errors = []
        for q in result_queues:
            res = q.get()
            if res is not None:
                errors.append(res)
        
        if errors:
            raise errors[0]

    yield run_task

    for q in task_queues: q.put(None)
    for p in processes: 
        p.join(timeout=5)
        if p.is_alive(): p.terminate()
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

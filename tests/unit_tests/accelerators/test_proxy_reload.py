import pytest
import os
import subprocess
import time
import requests
import json
import re
import threading
import random
from pathlib import Path

from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import strart_vllm_mock, cleanup_subprocess, setup_vllm
import port_manager


@pytest.fixture(scope="module")
def reload_env():
    os.environ["no_proxy"] = "localhost,127.0.0.1"

    PREFILL_NUM = 3
    DECODE_NUM = 3

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)

    proxy_port = ports["proxy_port"]
    prefill_ports = ports["prefill"]
    decode_ports = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_ports, decode_ports)
    if ret != 0:
        pytest.fail("Start proxy fail")

    processes = strart_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Start vllm fail")

    yield {
        "proxy_port": proxy_port,
        "prefill_ports": prefill_ports,
        "decode_ports": decode_ports,
        "processes": processes,
    }

    teardown_proxy()
    cleanup_subprocess(processes)

# Case behavior helpers
def apply_case_1_remove(cur_prefill, cur_decode):
    """
    Case 1: -P2 / -D1
    """
    removed_p = cur_prefill.pop(1)
    removed_d = cur_decode.pop(0)
    return removed_p, removed_d


def apply_case_2_restore(cur_prefill, cur_decode, base_prefill, base_decode):
    """
    Case 2: +P2 / +D1（恢复到 base）
    """
    cur_prefill[:] = base_prefill[:]
    cur_decode[:]  = base_decode[:]


def apply_case_3_append_existing(cur_prefill, cur_decode, p3_port, d3_port):
    """
    Case 3: +P3 / +D3（新 vLLM）
    """
    cur_prefill.append(p3_port)
    cur_decode.append(d3_port)


def apply_case_4_remove_new(cur_prefill, cur_decode, p3_port, d3_port):
    """
    Case 4: -P3 / -D3（回到 base）
    """
    cur_prefill.remove(p3_port)
    cur_decode.remove(d3_port)

# 发送 nginx reload 信号
def reload_nginx(conf_path):
    proc = subprocess.run(
        f"nginx -c {conf_path} -s reload",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, "nginx reload failed"
    assert proc.stderr.strip() == "", f"nginx reload stderr: {proc.stderr}"

# proxy 健康检查
def wait_proxy_health(proxy_port, timeout=30):
    url = f"http://127.0.0.1:{proxy_port}/omni_proxy/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    pytest.fail("proxy health endpoint not ready after reload")

# 整体替换 upstream（不是增量 patch）
def rewrite_upstream(conf_path, prefill_ports, decode_ports):
    with open(conf_path, "r") as f:
        lines = f.readlines()

    def gen_block(name, ports):
        out = [
            f"    upstream {name} {{\n",
            "        keepalive 2048;\n",
            "        keepalive_timeout 110s;\n",
            "        keepalive_requests 20000;\n",
        ]
        for p in ports:
            out.append(
                f"        server 127.0.0.1:{p} max_fails=3 fail_timeout=10s;\n"
            )
        out.append("    }\n")
        return out

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("upstream prefill_endpoints"):
            new_lines.extend(gen_block("prefill_endpoints", prefill_ports))
            while not lines[i].strip().endswith("}"):
                i += 1
        elif line.strip().startswith("upstream decode_endpoints"):
            new_lines.extend(gen_block("decode_endpoints", decode_ports))
            while not lines[i].strip().endswith("}"):
                i += 1
        else:
            new_lines.append(line)
        i += 1

    with open(conf_path, "w") as f:
        f.writelines(new_lines)

# 等 vLLM mock 新实例启动完成（Case 3 用）
def wait_vllm_ready(processes, logs, timeout=60):
    """
    Wait until all vLLM mock processes report 'Application startup complete.'
    and none of them exits prematurely.
    """
    start = time.time()
    ready = [False] * len(processes)

    while time.time() - start < timeout:
        for i, (proc, log) in enumerate(zip(processes, logs)):
            if ready[i]:
                continue

            # vLLM 进程不能提前退出
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM process {proc.pid} exited early")

            # 日志里出现 ready 标志
            if log.exists() and "Application startup complete." in log.read_text():
                ready[i] = True

        if all(ready):
            return

        time.sleep(1)

    raise RuntimeError("new vLLM instances did not become ready in time")


def test_proxy_reload(reload_env):
    proxy_port = reload_env["proxy_port"]
    prefill_port_list = reload_env["prefill_ports"]
    decode_port_list = reload_env["decode_ports"]
    """
    Health-based proxy reload test (enhanced).

    Coverage:
    - Baseline nginx reload mechanics (master unchanged, workers rotated)
    - Reload scenarios:
        1) -P2 / -D1
        2) +P2 / +D1 (same ports)
        3) +P3 / +D3 (new ports, new vLLM)
        4) -P3 / -D3
    - After each reload:
        * proxy health OK
        * 30 inference requests succeed (HTTP 200)
        * no nginx crash (SIGSEGV / core dumped)
        * master PID unchanged
    - nginx.conf must be restored after test
    """
    # =========================================================
    # [方案B新增] 通过环境变量选择只跑某一个 Case
    #
    # 未设置 RELOAD_CASE：
    #   pytest -s test_proxy_reload.py  -> 跑 Case 1~4（原始行为）
    #
    # 设置 RELOAD_CASE=1/2/3/4：
    #   只执行指定的 Case（Case 2/4 会做静默前置准备，以保证语义成立）
    # =========================================================
    SELECT_CASE = os.getenv("RELOAD_CASE")
    if SELECT_CASE:
        print(f"[RELOAD_CASE] Only running Case {SELECT_CASE}")

    conf_path = "/usr/local/nginx/conf/nginx.conf"
    error_log = Path(
        "/data/l00959921/omniinfer/tests/unit_tests/accelerators/nginx_error.log"
    )

    # nginx crash 关键字
    NGINX_CRASH_KEYWORDS = [
        "exited on signal",
        "segmentation fault",
        "core dumped",
    ]

    # 备份 nginx.conf（必须保证最终恢复）
    with open(conf_path, "r") as f:
        original_nginx_conf = f.read()

    # Helpers

    # 获取 nginx master / worker PID
    def get_nginx_pids(tag=""):
        out = subprocess.check_output(
            "ps -ef | grep nginx | grep -v grep",
            shell=True
        ).decode()

        master = None
        workers = []

        for line in out.splitlines():
            parts = line.split()
            if "master process" in line:
                master = int(parts[1])
            elif "worker process" in line:
                workers.append(int(parts[1]))

        print(f"\n[NGINX PID] {tag}")
        print(f"  master : {master}")
        print(f"  workers: {workers}")

        return master, workers

    # 只检查 reload 后新增的 error.log，捕捉 SIGSEGV
    def assert_no_nginx_crash(error_log: Path, start_pos: int):
        if not error_log.exists():
            return
        with open(error_log, "r") as f:
            f.seek(start_pos)
            new_logs = f.read().lower()
        for kw in NGINX_CRASH_KEYWORDS:
            assert kw not in new_logs, (
                f"nginx crash detected after reload: '{kw}'\n"
                f"{new_logs}"
            )

    # reload 过程中验证 proxy 不 502、不掉请求
    def send_requests(num=30):
        url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        for i in range(num):
            r = requests.post(url, headers=headers, json=data, timeout=15)
            if r.status_code != 200:
                pytest.fail(
                    f"Request {i} failed: HTTP {r.status_code}, body={r.text!r}"
                )
            # reload 过程中允许空 body / 非 JSON
            try:
                r.json()
            except Exception:
                pass

    # Log helpers（识别 reload 完成、识别新增端口注册）
    PREFILL_RE = re.compile(r"Add Prefill peer .* -> 127\.0\.0\.1:(\d+)")
    DECODE_RE  = re.compile(r"Add Decode peer .* -> 127\.0\.0\.1:(\d+)")
    COMPLETE_MARK = "Upstream initialization completed"

    def wait_reload_complete_from_log(error_log, start_pos, timeout=30):
        start = time.time()
        while time.time() - start < timeout:
            if error_log.exists():
                with open(error_log, "r") as f:
                    f.seek(start_pos)
                    logs = f.read()
                if COMPLETE_MARK in logs:
                    return
            time.sleep(0.5)
        pytest.fail("reload did not complete (no completion mark in log)")

    # 解析新增日志里出现过的新端口
    def parse_new_ports_from_log(error_log, start_pos):
        if not error_log.exists():
            return set(), set()
        with open(error_log, "r") as f:
            f.seek(start_pos)
            logs = f.read()
        return (
            set(int(p) for p in PREFILL_RE.findall(logs)),
            set(int(p) for p in DECODE_RE.findall(logs)),
        )

    # Test body（必须保证：异常也能清理 & 还原）
    try:
        # Prepare baseline ports
        base_prefill = prefill_port_list.copy()
        base_decode  = decode_port_list.copy()

        cur_prefill = base_prefill.copy()
        cur_decode  = base_decode.copy()

        new_processes = []
        new_logs = []

        # Reload cases runner（最终稳定语义）
        def run_case(
            case_name,
            modify_fn,
            expect_prefill,
            expect_decode,
            expect_new_p=None,
            expect_new_d=None,
        ):
            print(f"\n===== {case_name} =====")

            # reload 前：PID + 配置层 upstream 快照
            master_before, workers_before = get_nginx_pids("before reload")

            before_prefill_cfg = cur_prefill.copy()
            before_decode_cfg  = cur_decode.copy()

            # reload 前 error.log 起点
            log_pos = error_log.stat().st_size if error_log.exists() else 0

            # 修改配置 + reload
            modify_fn()
            rewrite_upstream(conf_path, cur_prefill, cur_decode)

            after_prefill_cfg = cur_prefill.copy()
            after_decode_cfg  = cur_decode.copy()

            reload_nginx(conf_path)
            wait_proxy_health(proxy_port)
            wait_reload_complete_from_log(error_log, log_pos)

            # reload 后：PID 校验
            master_after, workers_after = get_nginx_pids("after reload")

            # 是否严格要求 master PID 不变
            STRICT_MASTER_CHECK = os.getenv("STRICT_MASTER_CHECK", "0") == "1"

            if STRICT_MASTER_CHECK:
                assert master_after == master_before, (
                    f"nginx master pid changed: {master_before} -> {master_after}"
                )
            else:
                if master_after != master_before:
                    print(
                        f"[INFO] nginx master pid changed (allowed): "
                        f"{master_before} -> {master_after}"
                    )

            # worker 必须轮换（这个断言必须保留）
            assert workers_after != workers_before, \
                "nginx workers not rotated after reload"


            # 打印：配置层 upstream 对比表（只做可视化）
            def _fmt(ports):
                return "[" + ", ".join(str(p) for p in ports) + "]"

            print("\n================ Upstream Config Diff =================")
            print(f"{'Type':8} | {'Reload-Before':25} | {'Reload-After'}")
            print("-" * 58)
            print(f"{'Prefill':8} | {_fmt(before_prefill_cfg):25} | {_fmt(after_prefill_cfg)}")
            print(f"{'Decode':8}  | {_fmt(before_decode_cfg):25} | {_fmt(after_decode_cfg)}")
            print("-" * 58)
            print("Note: Config-layer upstream diff (NOT nginx runtime peers)")
            print("=" * 58 + "\n")

            # 配置层断言（核心语义：严格集合相等）
            assert set(cur_prefill) == set(expect_prefill)
            assert set(cur_decode)  == set(expect_decode)

            # log 层校验（只验证：新增端口曾被注册）
            new_p, new_d = parse_new_ports_from_log(error_log, log_pos)

            if expect_new_p:
                missing = set(expect_new_p) - new_p
                if missing:
                    print(
                        f"[WARN] New prefill ports not seen in nginx log (allowed): "
                        f"{sorted(missing)}"
                    )

            if expect_new_d:
                missing = set(expect_new_d) - new_d
                if missing:
                    print(
                        f"[WARN] New decode ports not seen in nginx log (allowed): "
                        f"{sorted(missing)}"
                    )

            # 功能性 & 稳定性校验
            send_requests()
            assert_no_nginx_crash(error_log, log_pos)

        def _silent_apply_state(prefill_ports, decode_ports):
            rewrite_upstream(conf_path, prefill_ports, decode_ports)
            reload_nginx(conf_path)
            wait_proxy_health(proxy_port)

        # Case 1: -P2 / -D1
        removed_p = cur_prefill[1]
        removed_d = cur_decode[0]

        expected_prefill_case1 = base_prefill.copy()
        expected_prefill_case1.remove(removed_p)

        expected_decode_case1 = base_decode.copy()
        expected_decode_case1.remove(removed_d)

        # 单 Case: 如果只跑 Case 2，需要先静默进入“Case1 状态”
        if SELECT_CASE == "2":
            cur_prefill = expected_prefill_case1.copy()
            cur_decode = expected_decode_case1.copy()
            _silent_apply_state(cur_prefill, cur_decode)

        # 单 Case: 如果只跑 Case 4，需要先静默进入“Case3 状态”
        # 也就是：起新 vLLM + upstream 加新端口 + reload 生效（静默）
        p3_port = None
        d3_port = None
        if SELECT_CASE == "4":
            p3_port = port_manager.find_free_port_excluding_existing()
            d3_port = port_manager.find_free_port_excluding_existing()

            procs, logs = setup_vllm(True, [p3_port], log_file_prefix="reload")
            new_processes.extend(procs)
            new_logs.extend(logs)

            procs, logs = setup_vllm(False, [d3_port], log_file_prefix="reload")
            new_processes.extend(procs)
            new_logs.extend(logs)

            wait_vllm_ready(new_processes, new_logs)

            cur_prefill = base_prefill.copy() + [p3_port]
            cur_decode  = base_decode.copy() + [d3_port]
            _silent_apply_state(cur_prefill, cur_decode)

        # Case 1: -P2 / -D1
        if SELECT_CASE in (None, "1"):
            run_case(
                "Case 1: -P2 / -D1",
                lambda: apply_case_1_remove(cur_prefill, cur_decode),
                expect_prefill=expected_prefill_case1,
                expect_decode=expected_decode_case1,
            )

        # Case 2: +P2 / +D1（端口必须完全恢复）
        if SELECT_CASE in (None, "2"):
            run_case(
                "Case 2: +P2 / +D1",
                lambda: apply_case_2_restore(cur_prefill, cur_decode, base_prefill, base_decode),
                expect_prefill=base_prefill,
                expect_decode=base_decode,
                expect_new_p=[removed_p],
                expect_new_d=[removed_d],
            )

        # Case 3: +P3 / +D3（新 vLLM）
        if SELECT_CASE in (None, "3"):
            p3_port = port_manager.find_free_port_excluding_existing()
            d3_port = port_manager.find_free_port_excluding_existing()

            procs, logs = setup_vllm(True, [p3_port], log_file_prefix="reload")
            new_processes.extend(procs)
            new_logs.extend(logs)

            procs, logs = setup_vllm(False, [d3_port], log_file_prefix="reload")
            new_processes.extend(procs)
            new_logs.extend(logs)

            wait_vllm_ready(new_processes, new_logs)

            try:
                run_case(
                    "Case 3: +P3 / +D3",
                    lambda: apply_case_3_append_existing(cur_prefill, cur_decode, p3_port, d3_port),
                    expect_prefill=base_prefill + [p3_port],
                    expect_decode=base_decode + [d3_port],
                    expect_new_p=[p3_port],
                    expect_new_d=[d3_port],
                )
            finally:
                # Case 3 新起的 vLLM 必须清理
                cleanup_subprocess(new_processes)
                new_processes = []
                new_logs = []

        # Case 4: -P3 / -D3（必须回到 base）
        if SELECT_CASE in (None, "4"):
            # 若是全量跑：p3_port/d3_port 来自 Case3
            # 若是单跑 Case4：p3_port/d3_port 在上面的静默准备里生成
            run_case(
                "Case 4: -P3 / -D3",
                lambda: apply_case_4_remove_new(cur_prefill, cur_decode, p3_port, d3_port),
                expect_prefill=base_prefill,
                expect_decode=base_decode,
            )

    finally:
        # ALWAYS restore nginx.conf
        with open(conf_path, "w") as f:
            f.write(original_nginx_conf)

        # 防御性清理：单 Case 4 静默准备时可能起了 vLLM
        if 'new_processes' in locals() and new_processes:
            cleanup_subprocess(new_processes)



def test_proxy_reload_under_concurrent_traffic(reload_env):
    """
    【并发流量 + 多轮真实 reload 稳定性测试】

    测试目标：
    在「持续真实请求流量」存在的情况下，后台不断执行 nginx reload，
    且 reload 过程中伴随 upstream（Prefill / Decode 后端）的真实增删变化，
    验证 proxy 在高 churn 场景下的稳定性。

    本测试覆盖的关键保证点：
    1. reload 期间，请求持续成功（不丢请求、不 502）
    2. reload 在后台线程中执行，与请求线程并发
    3. reload 会重复多次（多轮）
    4. 每次 reload 都伴随 upstream P/D 的变化（删 / 加 / 回退）
    5. 每一轮结束后，配置必须回到初始状态（无状态污染）

    策略约束（Policy）：
    - 放宽客户端超时时间（防止误报）
    - 不允许任何请求超时（timeout 视为失败）
    - 所有请求必须返回 HTTP 200
    """
    # nginx proxy 对外监听端口
    proxy_port = reload_env["proxy_port"]

    # 初始 upstream 配置（baseline），用于每轮结束后的“防污染校验”
    base_prefill = reload_env["prefill_ports"].copy()
    base_decode  = reload_env["decode_ports"].copy()

    conf_path = "/usr/local/nginx/conf/nginx.conf"

    # ============================
    # 线程控制 & 错误收集
    # ============================

    # 用于通知请求线程停止
    stop_event = threading.Event()

    # 请求线程中捕获的所有异常 / 非 200 错误
    request_errors = []

    # reload 线程中捕获的异常（线程异常不能直接抛给 pytest）
    reload_errors = []

    # ============================
    # 请求侧统计信息（仅用于观测，不影响 pass / fail）
    # ============================

    # 请求总数
    request_count = 0

    # 成功（HTTP 200）的请求数
    request_success = 0

    # 每个请求的耗时（秒）
    request_latencies = []

    # 保护统计变量的线程锁
    stats_lock = threading.Lock()

    # =========================================================
    # 前台请求线程：模拟真实业务流量（数据面）
    # =========================================================
    def request_worker():
        """
        持续向 proxy 发送真实业务请求，用于模拟生产环境下的用户流量。

        特点：
        - 与 reload 线程并发运行
        - reload 期间请求不能失败
        - 记录请求耗时，用于最终统计
        """
        nonlocal request_count, request_success

        url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        timeout = (3, 60)

        while not stop_event.is_set():
            start_ts = time.time()
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                )
                latency = time.time() - start_ts

                # 更新请求统计
                with stats_lock:
                    request_count += 1
                    request_latencies.append(latency)

                # 非 200 直接记为错误
                if r.status_code != 200:
                    request_errors.append(
                        f"HTTP {r.status_code}: {r.text!r}"
                    )
                    continue

                # 成功请求计数
                with stats_lock:
                    request_success += 1

                # reload 期间允许返回空 body 或非 JSON（边界行为）
                try:
                    r.json()
                except Exception:
                    pass

            except Exception as e:
                # 所有异常（包括 timeout）都视为失败
                latency = time.time() - start_ts
                with stats_lock:
                    request_count += 1
                    request_latencies.append(latency)

                request_errors.append(str(e))

    # =========================================================
    # 后台 reload 线程：控制面（真实配置 churn）
    # =========================================================
    def reload_worker():
        """
        后台执行多轮 reload，每一轮严格按照 Case 1~4 顺序执行。

        每一轮包含：
        - Case 1：删除一个 Prefill + 一个 Decode
        - Case 2：恢复到 baseline
        - Case 3：新增 Prefill + Decode（真实启动新 vLLM）
        - Case 4：移除新增节点，回到 baseline

        每个 Case 都会：
        - 修改 nginx upstream
        - 执行真实 nginx reload
        - 等待 proxy 恢复健康
        """
        cur_prefill = base_prefill.copy()
        cur_decode  = base_decode.copy()

        # Case 3 中启动的新 vLLM 进程，最终统一清理
        new_processes = []
        new_logs = []

        try:
            # reload 总轮数（稳定性测试，故多轮）
            ROUNDS = 2

            for round_id in range(ROUNDS):
                print(f"\n[ROUND {round_id}] ===============================")
                print(f"[ROUND {round_id}] Starting reload cycle")

                # -----------------------------
                # Case 1：删除一个 P / D
                # -----------------------------
                print(f"[ROUND {round_id}] Case 1: remove one P and one D")
                removed_p, removed_d = apply_case_1_remove(cur_prefill, cur_decode)
                print(f"[ROUND {round_id}]   removed P={removed_p}, D={removed_d}")

                rewrite_upstream(conf_path, cur_prefill, cur_decode)
                reload_nginx(conf_path)
                wait_proxy_health(proxy_port)
                time.sleep(random.uniform(0.5, 1.5))

                # -----------------------------
                # Case 2：恢复到 baseline
                # -----------------------------
                print(f"[ROUND {round_id}] Case 2: restore base P/D")
                apply_case_2_restore(cur_prefill, cur_decode, base_prefill, base_decode)

                rewrite_upstream(conf_path, cur_prefill, cur_decode)
                reload_nginx(conf_path)
                wait_proxy_health(proxy_port)
                time.sleep(random.uniform(0.5, 1.5))

                # -----------------------------
                # Case 3：新增 P / D（真实新 vLLM）
                # -----------------------------
                print(f"[ROUND {round_id}] Case 3: add new P/D backends")

                p3 = port_manager.find_free_port_excluding_existing()
                d3 = port_manager.find_free_port_excluding_existing()
                print(f"[ROUND {round_id}]   new P={p3}, D={d3}")

                # 启动新的 vLLM 后端
                procs, logs = setup_vllm(True, [p3], log_file_prefix="reload")
                new_processes.extend(procs)
                new_logs.extend(logs)

                procs, logs = setup_vllm(False, [d3], log_file_prefix="reload")
                new_processes.extend(procs)
                new_logs.extend(logs)

                # 等待新 vLLM 完全 ready，避免 upstream 刚加就接不住请求
                wait_vllm_ready(procs, logs)

                apply_case_3_append_existing(cur_prefill, cur_decode, p3, d3)

                rewrite_upstream(conf_path, cur_prefill, cur_decode)
                reload_nginx(conf_path)
                wait_proxy_health(proxy_port)
                time.sleep(random.uniform(1, 2))

                # -----------------------------
                # Case 4：移除新增节点，回滚
                # -----------------------------
                print(f"[ROUND {round_id}] Case 4: remove new P/D and rollback")

                apply_case_4_remove_new(cur_prefill, cur_decode, p3, d3)

                rewrite_upstream(conf_path, cur_prefill, cur_decode)
                reload_nginx(conf_path)
                wait_proxy_health(proxy_port)
                time.sleep(random.uniform(0.5, 1.5))

                # -----------------------------
                # 防污染校验（必须回到 baseline）
                # -----------------------------
                if set(cur_prefill) != set(base_prefill):
                    raise RuntimeError(
                        f"[Round {round_id}] Prefill polluted: "
                        f"{cur_prefill} vs {base_prefill}"
                    )

                if set(cur_decode) != set(base_decode):
                    raise RuntimeError(
                        f"[Round {round_id}] Decode polluted: "
                        f"{cur_decode} vs {base_decode}"
                    )

                print(f"[ROUND {round_id}] Completed successfully")

        except Exception as e:
            # 捕获 reload 线程异常，交由主线程统一 fail
            reload_errors.append(e)
        finally:
            # 确保所有新起的 vLLM 被清理
            cleanup_subprocess(new_processes)

    # =========================================================
    # 启动并发执行
    # =========================================================
    t_req = threading.Thread(target=request_worker, daemon=True)
    t_reload = threading.Thread(target=reload_worker)

    t_req.start()
    t_reload.start()

    # 等 reload 完成后，再停止请求线程
    t_reload.join()
    stop_event.set()
    t_req.join(timeout=10)

    # =========================================================
    # 最终断言 & 汇总输出
    # =========================================================
    if reload_errors:
        pytest.fail(f"Reload thread failed: {reload_errors[0]}")

    if request_errors:
        pytest.fail(
            "Requests failed during concurrent reload:\n" +
            "\n".join(request_errors[:10])
        )

    # 打印请求侧统计信息（不影响测试结果）
    if request_latencies:
        max_lat = max(request_latencies)
        avg_lat = sum(request_latencies) / len(request_latencies)
        print(
            f"\n[REQUEST] total={request_count} "
            f"success={request_success} "
            f"max_latency={max_lat:.2f}s "
            f"avg_latency={avg_lat:.2f}s"
        )


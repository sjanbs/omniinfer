import sys
import subprocess
from pathlib import Path

CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../../../omni/accelerators/sched/omni_proxy/omni_proxy.sh"

def generate_proxy_endpoints(port: int, num: int) -> str:
    return ",".join(f"127.0.0.1:{port + i}" for i in range(num))

def setup_proxy(proxy_port=7000, prefill_num=1, prefill_base_port=8000, decode_num=1, decode_base_port=9000):
    prefill_list = generate_proxy_endpoints(prefill_base_port, prefill_num)
    decode_list = generate_proxy_endpoints(decode_base_port, decode_num)
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", "/usr/local/nginx/conf/nginx.conf",
            "--core-num", "1",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", "/usr/local/nginx/logs/error.log",
            "--log-level", "info",
            "--access-log-file", "/usr/local/nginx/logs/access.log",
            "--stream-ops", "add",
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

def teardown_proxy():
    try:
        cmd = [
            "bash", proxy_script_path,
            "--stop",
        ]
        print(f"\n[TEARDOWN] Stopping proxy with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[TEARDOWN] Script succeeded. Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Teardown script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        pytest.fail(error_msg)


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        setup_proxy()
    elif len(args) == 5:
        try:
            numeric_args = []
            for arg in args:
                numeric_args.append(int(arg))
            setup_proxy(*numeric_args)
        except ValueError as e:
            print(f"Error: All four arguments must be valid numbers. Got: {args}")
            print("Usage: python script.py <proxy_port> <prefill_num> <prefill_base_port> <decode_num> <decode_base_port>")
            sys.exit(1)
    elif len(args) == 1 and args[0] == "stop":
        teardown_proxy()
    else:
        print(f"Error: Invalid arguments: {args}")
        print("Usage:")
        print("  python run_proxy.py <proxy_port> <prefill_num> <prefill_base_port> <decode_num> <decode_base_port>  # Start with 5 numbers")
        print("  python run_proxy.py stop                         # Stop/cleanup")
        sys.exit(1)

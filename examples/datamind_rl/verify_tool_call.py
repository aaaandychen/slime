"""Quick verification: model tool calling with sglang + Claude Code.

Prerequisite: model at /mnt/cephfs/chenzhenyang/models/Qwen2.5-Coder-14B-Instruct
"""

import subprocess, os, json, tempfile, sys

MODEL = "/mnt/cephfs/chenzhenyang/models/Qwen2.5-Coder-14B-Instruct"

def check_model():
    import glob
    shards = glob.glob(f"{MODEL}/*.safetensors") + glob.glob(f"{MODEL}/._____temp/*.safetensors")
    if not shards:
        print("ERROR: No safetensors found. Model not ready.")
        sys.exit(1)
    total = sum(os.path.getsize(s) for s in shards)
    print(f"Model shards: {len(shards)}, total: {total/1e9:.1f}GB")
    # Qwen2.5-Coder-14B is ~28GB
    if total < 25e9:
        print("WARNING: Model may be incomplete (expecting ~28GB).")
    else:
        print("Model looks complete!")

def test_sglang_generate():
    """Start sglang and test tool calling."""
    import requests, time, sys

    # Launch sglang
    port = 30000
    print(f"Launching sglang on port {port}...")
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tp-size", "1",
        "--mem-fraction-static", "0.85",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(30)

    # Check health
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                print("sglang ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("ERROR: sglang not ready")
        proc.terminate()
        sys.exit(1)

    # Test tool call
    prompt = """<|im_start|>system
You are a data analyst. Use the Read tool to read files, Bash to run commands.
<|im_end|>
<|im_start|>user
Read the file data.csv and tell me the number of rows.<|im_end|>
<|im_start|>assistant
"""

    print("\n=== Testing tool call ===")
    resp = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0.7, "max_new_tokens": 200},
        },
    )
    print(resp.json()["text"][:500])

    proc.terminate()
    print("\nDone.")

if __name__ == "__main__":
    check_model()
    # Uncomment to test sglang:
    # test_sglang_generate()

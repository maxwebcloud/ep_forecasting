
import subprocess
from concurrent.futures import ThreadPoolExecutor


"""
multi_process.py

- Launch each forecasting model in its own Python subprocess.
- Run GPU‑configured models sequentially 
- Run CPU‑configured models in parallel
- Each subprocess calls run_all_models.py with --mode and --device flags.
"""

GPU_MODELS = ["lstm", "slstm"]
CPU_MODELS = ["rnn", "plstm"]

def run_job(mode: str, device: str):
    print(f"→ Starte {mode.upper()} auf {device} …")
    subprocess.run(
        ["python", "run_all_models.py", "--mode", mode, "--device", device],
        check=True
    )

def main():
    # 1) CPU‑Jobs parallel starten
    with ThreadPoolExecutor(max_workers=2) as exe:
        cpu_futures = [exe.submit(run_job, m, "cpu") for m in CPU_MODELS]

        # 2) Parallel dazu: GPU‑Jobs seriell
        for mode in GPU_MODELS:
            run_job(mode, "mps")

        # 3) Warten, bis alle CPU‑Jobs fertig sind
        for f in cpu_futures:
            f.result()

if __name__ == "__main__":
    main()
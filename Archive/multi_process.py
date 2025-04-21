#!/usr/bin/env python3
import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

"""
multi_process.py

- Launch each forecasting model in its own Python subprocess.
- Run GPU‑configured models sequentially.
- Run CPU‑configured models in parallel.
- Stream and display all output to terminal, while simultaneously logging to gesamtoutput.txt.
- Append a runtime summary at the end.
"""

GPU_MODELS = ["lstm", "slstm"]  # laufend sequenziell auf MPS
CPU_MODELS = ["rnn", "plstm"]   # laufend parallel auf CPU


def run_job(mode: str, device: str, log_f):
    """Startet ein Subprozess, streamt stdout/stderr zu Terminal und Log."""
    header = f"→ Starte {mode.upper()} auf {device} (Start: {datetime.now():%H:%M:%S})…\n"
    print(header, end='')
    log_f.write(header)
    t0 = time.time()
    cmd = ["python", "run_all_models.py", "--mode", mode, "--device", device]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
        log_f.write(line)
    proc.wait()
    duration = time.time() - t0
    summary = f"→ Dauer {mode.upper()}: {duration/60:.2f} Minuten\n\n"
    print(summary, end='')
    log_f.write(summary)
    return mode, duration


def main():
    out_dir = "model_metrics_overview"
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, "gesamtoutput.txt")

    with open(logfile, "w") as f:
        f.write(f"Run started at: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        runtimes = {}

        # CPU‑Jobs parallel
        with ThreadPoolExecutor(max_workers=2) as execr:
            futures = {execr.submit(run_job, m, "cpu", f): m for m in CPU_MODELS}

            # GPU‑Jobs seriell
            for mode in GPU_MODELS:
                m, dur = run_job(mode, "mps", f)
                runtimes[m] = dur

            # Warte auf CPU‑Jobs
            for fut in as_completed(futures):
                m, dur = fut.result()
                runtimes[m] = dur

        # Laufzeit‑Zusammenfassung
        f.write("=== Runtime Summary Multi Process ===\n")
        print("=== Runtime Summary Multi Process ===")
        total = 0.0
        for mode in GPU_MODELS + CPU_MODELS:
            rt = runtimes.get(mode, 0.0)
            line = f"Model {mode.upper():6s} → {rt/60:.2f} Minuten\n"
            print(line, end='')
            f.write(line)
            total += rt
        final = f"\nTotal runtime across all models: {total/60:.2f} Minuten\n"
        print(final, end='')
        f.write(final)

    print(f"\nAlle Logs und Summaries wurden in '{logfile}' gespeichert.")


if __name__ == "__main__":
    main()

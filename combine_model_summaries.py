from pathlib import Path
import re

input_folder = Path("model_metrics_overview")
output_file = input_folder / "combined_summary.txt"

# Regex zum Finden von Zeilen wie: "Total runtime across all models: 12.34 minutes"
RUNTIME_RE = re.compile(r"Total runtime across all models:\s*([0-9.]+)\s*minutes")

summary_files = sorted(input_folder.glob("summary_*.txt"))

if not summary_files:
    raise SystemExit(f"No summary_*.txt files found in {input_folder}")

total_runtime = 0.0

with output_file.open("w", encoding="utf-8") as out:
    for file in summary_files:
        out.write(f"\n===== {file.name} =====\n\n")
        text = file.read_text(encoding="utf-8")
        out.write(text + "\n")

        match = RUNTIME_RE.search(text)
        if match:
            total_runtime += float(match.group(1))

    out.write(f"\nAggregated total runtime across all models: {total_runtime:.2f} minutes\n")

print(f"✅ Merged {len(summary_files)} files ➜ {output_file.resolve()}")
print(f"⏱️  Aggregated total runtime: {total_runtime:.2f} minutes")
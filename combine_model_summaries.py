"""
This script iterates through all 'summary_*.txt' files in the 'model_metrics_overview' folder,
merges their content into one output file, and extracts runtime values from each file to compute
the aggregated total runtime across all models.
"""

from pathlib import Path
import re

folder = Path("model_metrics_overview")
files = sorted(folder.glob("summary_*.txt"))
if not files:
    raise SystemExit("No summary files found.")

runtime_re = re.compile(r"Runtime:\s*([\d.]+)\s*minutes")
total = 0.0

with (folder / "combined_summary.txt").open("w", encoding="utf-8") as out:
    for f in files:
        text = f.read_text(encoding="utf-8")
        out.write(text + "\n")
        if match := runtime_re.search(text):
            total += float(match.group(1))
    out.write(f"\nAggregated total runtime across all models: {total:.2f} minutes\n")

print(f"✅ Merged {len(files)} files ➜ {folder / 'combined_summary.txt'}")
print(f"⏱️  Aggregated total runtime: {total:.2f} minutes")
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "prefill_tokens_per_second": "Prefill throughput (tokens/s)",
    "decode_tokens_per_second": "Decode throughput (tokens/s)",
    "time_to_first_token_ms": "Time to first token (ms)",
    "peak_accelerator_memory_gb": "Peak accelerator memory (GB)",
}


def create_plots(csv_path: str, output_dir: str) -> list[Path]:
    data = pd.read_csv(csv_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for metric, ylabel in METRICS.items():
        if metric not in data or data[metric].dropna().empty:
            continue
        fig, axis = plt.subplots(figsize=(7, 4.5))
        group_columns = [c for c in ("size", "precision", "accelerator_name") if c in data]
        for key, group in data.groupby(group_columns, dropna=False):
            group = group.sort_values("requested_context_tokens")
            label = " / ".join(map(str, key if isinstance(key, tuple) else (key,)))
            axis.plot(group["requested_context_tokens"], group[metric], marker="o", label=label)
        axis.set_xlabel("Requested context length (tokens)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Context length vs {ylabel.lower()}")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        path = destination / f"context_vs_{metric}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        outputs.append(path)
    return outputs


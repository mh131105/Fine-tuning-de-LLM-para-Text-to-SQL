from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import append_jsonl, save_json


def setup_logger(name: str, output_dir: str | Path) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(Path(output_dir) / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_environment_snapshot(
    output_dir: str | Path,
    seed: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "model_name": model_name,
        "packages": {
            name: _package_version(name)
            for name in ["torch", "transformers", "datasets", "peft", "trl", "deepeval", "bitsandbytes"]
        },
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
    }
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            snapshot["gpu_name"] = properties.name
            snapshot["gpu_vram_gb"] = round(properties.total_memory / (1024**3), 2)
    except Exception as exc:
        snapshot["torch_probe_error"] = str(exc)
    save_json(Path(output_dir) / "environment.json", snapshot)
    return snapshot


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    save_json(path, metrics)


def write_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)


def write_summary_markdown(
    experiment_dir: str | Path,
    spider_metrics: dict[str, Any],
    mmlu_metrics: dict[str, Any],
    deltas: dict[str, Any] | None = None,
) -> Path:
    output = Path(experiment_dir)
    deltas = deltas or {}
    lines = [
        "# Benchmark Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Spider Execution Accuracy | {spider_metrics.get('execution_accuracy', 0.0):.4f} |",
        f"| Spider Total | {spider_metrics.get('total', 0)} |",
        f"| MMLU Accuracy | {mmlu_metrics.get('accuracy', 0.0):.4f} |",
        f"| MMLU Total | {mmlu_metrics.get('total', 0)} |",
    ]
    if deltas:
        lines.extend(
            [
                f"| Delta Spider vs Base | {deltas.get('spider_accuracy_delta', 0.0):+.4f} |",
                f"| Delta MMLU vs Base | {deltas.get('mmlu_accuracy_delta', 0.0):+.4f} |",
            ]
        )
    lines.extend(["", "## MMLU by Category", "", "| Category | Accuracy | Correct | Total |", "| --- | ---: | ---: | ---: |"])
    for category, values in sorted((mmlu_metrics.get("by_category") or {}).items()):
        lines.append(
            f"| {category} | {values.get('accuracy', 0.0):.4f} | {values.get('correct', 0)} | {values.get('total', 0)} |"
        )
    path = output / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

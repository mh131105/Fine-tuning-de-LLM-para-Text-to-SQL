from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SpiderSource = Literal["auto", "none", "local", "zip", "hf"]
SPIDER_REQUIRED_FILES = ("train_spider.json", "dev.json", "tables.json")


@dataclass(frozen=True)
class SpiderDataSourceResult:
    data_dir: Path
    source: str
    source_path: Path | None = None
    message: str = ""


def has_spider_raw_files(path: str | Path) -> bool:
    root = Path(path)
    return all((root / filename).exists() for filename in SPIDER_REQUIRED_FILES) and (root / "database").exists()


def find_spider_root(path: str | Path, max_depth: int = 5) -> Path | None:
    root = Path(path)
    if has_spider_raw_files(root):
        return root
    if not root.exists() or not root.is_dir():
        return None
    root_depth = len(root.parts)
    for candidate in sorted(root.rglob("train_spider.json")):
        parent = candidate.parent
        if len(parent.parts) - root_depth > max_depth:
            continue
        if has_spider_raw_files(parent):
            return parent
    return None


def ensure_spider_raw_data(
    data_dir: str | Path,
    source: SpiderSource = "auto",
    source_path: str | Path | None = None,
    hf_repo: str = "dreamerdeo/multispider",
    cache_dir: str | Path = ".cache/datasets/spider",
    force: bool = False,
) -> SpiderDataSourceResult:
    destination = Path(data_dir)
    if has_spider_raw_files(destination) and not force:
        return SpiderDataSourceResult(destination, "existing", destination, "Spider raw files already exist.")

    if source == "none":
        return SpiderDataSourceResult(destination, "none", None, "Spider raw acquisition skipped.")

    if source_path is None:
        env_source = os.environ.get("SPIDER_SOURCE_PATH")
        if env_source:
            source_path = env_source

    if source in {"auto", "local", "zip"} and source_path:
        imported = import_spider_from_path(source_path, destination, force=force)
        return SpiderDataSourceResult(destination, "local", Path(source_path), f"Imported Spider from {imported}.")

    if source in {"auto", "hf"}:
        try:
            downloaded_root = download_spider_from_hf(hf_repo, cache_dir)
            imported = copy_spider_tree(downloaded_root, destination, force=force)
            return SpiderDataSourceResult(destination, "hf", downloaded_root, f"Downloaded Spider from {hf_repo}.")
        except Exception as exc:
            if source == "hf":
                raise
            raise FileNotFoundError(
                "Spider raw data was not found and automatic acquisition failed. "
                "Provide --source_path pointing to a Spider directory or archive, or set SPIDER_SOURCE_PATH. "
                f"Automatic Hugging Face error: {exc}"
            ) from exc

    raise ValueError(f"Unsupported Spider source: {source}")


def import_spider_from_path(source_path: str | Path, destination: str | Path, force: bool = False) -> Path:
    source = Path(source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Spider source path not found: {source}")
    if source.is_file():
        with tempfile.TemporaryDirectory(prefix="tp2_spider_extract_") as temp_dir:
            shutil.unpack_archive(str(source), temp_dir)
            root = find_spider_root(temp_dir)
            if root is None:
                raise FileNotFoundError(f"No Spider root found inside archive: {source}")
            return copy_spider_tree(root, destination, force=force)
    root = find_spider_root(source)
    if root is None:
        raise FileNotFoundError(f"No Spider root found under: {source}")
    return copy_spider_tree(root, destination, force=force)


def copy_spider_tree(source_root: str | Path, destination: str | Path, force: bool = False) -> Path:
    source = Path(source_root).resolve()
    target = Path(destination).resolve()
    if source == target and has_spider_raw_files(target):
        return target
    if target.exists():
        if not force:
            if has_spider_raw_files(target):
                return target
            raise FileExistsError(f"Destination exists but is not a valid Spider raw directory: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    if not has_spider_raw_files(target):
        raise FileNotFoundError(f"Copied source is not a valid Spider raw directory: {target}")
    return target


def download_spider_from_hf(repo_id: str, cache_dir: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[
            "dataset/spider/*",
            "dataset/spider/**",
            "spider/*",
            "spider/**",
            "train_spider.json",
            "dev.json",
            "tables.json",
            "database/**",
        ],
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    root = find_spider_root(local_dir)
    if root is None:
        raise FileNotFoundError(f"Downloaded dataset did not contain a Spider root: {repo_id}")
    return root

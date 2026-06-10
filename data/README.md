# Data layout

Place raw Spider files under `data/raw/spider/`, or let
`python -m scripts.prepare_spider` import/download them:

- `train_spider.json`
- `dev.json`
- `tables.json`
- `database/<db_id>/<db_id>.sqlite`

Supported ingestion sources:

- existing `data/raw/spider`;
- `--source_path` pointing to a Spider directory;
- `--source_path` pointing to a ZIP/TAR archive containing Spider;
- `--source hf --hf_repo dreamerdeo/multispider`.

Generated files go under `data/processed/` and are intentionally ignored by Git.

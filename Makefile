.PHONY: install test smoke smoke-mmlu prepare-spider prepare-mmlu train-a train-b bench-base bench-a bench-b

install:
	pip install -r requirements.txt

test:
	python -m pytest

smoke: test

smoke-mmlu:
	python -m scripts.prepare_mmlu --config configs/eval.yaml --mock --limit_per_category 2

prepare-spider:
	python -m scripts.prepare_spider --data_dir data/raw/spider --output_dir data/processed/spider

prepare-mmlu:
	python -m scripts.prepare_mmlu --config configs/eval.yaml

train-a:
	python -m scripts.train --config configs/train_lora_exp_a.yaml

train-b:
	python -m scripts.train --config configs/train_lora_exp_b.yaml

bench-base:
	python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/base

bench-a:
	python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_a

bench-b:
	python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_b

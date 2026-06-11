PYTHON ?= python3

.PHONY: install test smoke smoke-mmlu prepare-spider prepare-mmlu train-a train-b train-c bench-base bench-a bench-b bench-c diagnose-exp-c-spider-nostop diagnose-exp-c-ckpt875-spider-nostop diagnose-exp-c-nostop

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

smoke: test

smoke-mmlu:
	$(PYTHON) -m scripts.prepare_mmlu --config configs/eval.yaml --mock --limit_per_category 2

prepare-spider:
	$(PYTHON) -m scripts.prepare_spider --data_dir data/raw/spider --output_dir data/processed/spider

prepare-mmlu:
	$(PYTHON) -m scripts.prepare_mmlu --config configs/eval.yaml

train-a:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_a.yaml

train-b:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_b.yaml

train-c:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_c.yaml

bench-base:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/base

bench-a:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_a

bench-b:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_b

bench-c:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_c

diagnose-exp-c-spider-nostop:
	$(PYTHON) -m scripts.evaluate_spider --config configs/eval_spider_nostop.yaml --model_path outputs/exp_c --output_dir outputs/diagnostics/exp_c_spider_nostop

diagnose-exp-c-ckpt875-spider-nostop:
	$(PYTHON) -m scripts.evaluate_spider --config configs/eval_spider_nostop.yaml --model_path outputs/exp_c/checkpoint-875 --output_dir outputs/diagnostics/exp_c_ckpt875_spider_nostop

diagnose-exp-c-nostop: diagnose-exp-c-spider-nostop diagnose-exp-c-ckpt875-spider-nostop

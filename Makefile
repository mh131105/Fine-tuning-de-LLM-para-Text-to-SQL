PYTHON ?= python3
EXP_C_EPOCH1_CKPT ?= $(shell ls -d outputs/exp_c/checkpoint-* 2>/dev/null | sort -V | head -n 1)

.PHONY: install test smoke smoke-mmlu prepare-spider prepare-mmlu train-a train-b train-c train-d train-e train-f train-g train-h train-all train-extra train-3epoch bench-base bench-a bench-b bench-c bench-d bench-e bench-f bench-g bench-h bench-all bench-extra bench-3epoch diagnose-exp-c-spider-nostop diagnose-exp-c-epoch1-spider-nostop diagnose-exp-c-nostop

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

train-d:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_d.yaml

train-e:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_e.yaml

train-f:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_f.yaml

train-g:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_g.yaml

train-h:
	$(PYTHON) -m scripts.train --config configs/train_lora_exp_h.yaml

train-all: train-a train-b train-c train-d

train-extra: train-e train-f

train-3epoch: train-g train-h

bench-base:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/base

bench-a:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_a

bench-b:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_b

bench-c:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_c

bench-d:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_d

bench-e:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_e

bench-f:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_f

bench-g:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_g

bench-h:
	$(PYTHON) -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_h

bench-all: bench-base bench-a bench-b bench-c bench-d

bench-extra: bench-e bench-f

bench-3epoch: bench-g bench-h

diagnose-exp-c-spider-nostop:
	$(PYTHON) -m scripts.evaluate_spider --config configs/eval_spider_nostop.yaml --model_path outputs/exp_c --output_dir outputs/diagnostics/exp_c_spider_nostop

diagnose-exp-c-epoch1-spider-nostop:
	test -n "$(EXP_C_EPOCH1_CKPT)"
	$(PYTHON) -m scripts.evaluate_spider --config configs/eval_spider_nostop.yaml --model_path $(EXP_C_EPOCH1_CKPT) --output_dir outputs/diagnostics/exp_c_epoch1_spider_nostop

diagnose-exp-c-nostop: diagnose-exp-c-spider-nostop diagnose-exp-c-epoch1-spider-nostop

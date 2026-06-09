# TP2 NLP - Fine-Tuning Text-to-SQL

Pipeline reprodutivel para avaliar especializacao em Text-to-SQL com Spider e possivel perda de capacidade geral em MMLU apos fine-tuning LoRA/QLoRA do `Qwen/Qwen2.5-3B-Instruct`.

O repositorio foi desenhado para rodar localmente e no Google Colab sem caminhos fixos. Treino e inferencia reais baixam modelo grande e devem ser executados por voce em GPU.

## O que o projeto faz

- prepara Spider train/dev e schemas de bancos SQLite;
- prepara uma suite fixa de MMLU com 150 questoes;
- avalia modelo base em Spider dev por Execution Accuracy;
- treina dois experimentos LoRA/QLoRA no Spider train;
- avalia modelos fine-tuned em Spider dev e MMLU;
- calcula ganho no Spider e variacao no MMLU contra baseline;
- salva predicoes JSONL, metricas JSON, logs, ambiente e summaries Markdown;
- inclui testes unitarios, testes com mocks e smoke tests leves.

## Estrutura

```text
configs/             YAMLs de treino e avaliacao
custom_metrics/      ExecutionAccuracy compativel com DeepEval
scripts/             CLIs publicas
src/tp2/             logica principal
tests/               unitarios e smoke tests com mocks
notebooks/           runner Colab sem logica exclusiva
data/                dados locais ignorados pelo Git
outputs/             resultados locais ignorados pelo Git
reports/             notas para o relatorio final
```

## Instalacao

Recomendado: Python 3.10 a 3.12 no Colab.

```bash
pip install -r requirements.txt
```

Se o Colab ja tiver `torch`/CUDA instalados, voce pode manter a versao do ambiente e reinstalar as demais bibliotecas manualmente, mas o caminho reproduzivel padrao e o `requirements.txt`.

## Dados

Coloque o Spider bruto em:

```text
data/raw/spider/
├── train_spider.json
├── dev.json
├── tables.json
└── database/<db_id>/<db_id>.sqlite
```

O MMLU e baixado via Hugging Face Datasets por `scripts.prepare_mmlu`.

## Comandos principais

Preparar Spider:

```bash
python -m scripts.prepare_spider --data_dir data/raw/spider --output_dir data/processed/spider
```

Preparar MMLU 150:

```bash
python -m scripts.prepare_mmlu --config configs/eval.yaml
```

Treinar Experimento A:

```bash
python -m scripts.train --config configs/train_lora_exp_a.yaml
```

Treinar Experimento B:

```bash
python -m scripts.train --config configs/train_lora_exp_b.yaml
```

Avaliar baseline:

```bash
python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/base
```

Avaliar Exp A:

```bash
python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_a
```

Avaliar Exp B:

```bash
python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_b
```

## Smoke tests sem modelo grande

Esses comandos nao baixam Qwen nem treinam LoRA.

```bash
python -m pytest
python -m scripts.prepare_mmlu --config configs/eval.yaml --mock --limit_per_category 2
```

Tambem existe `--mock` nos avaliadores para testes locais de formato:

```bash
python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/base --mock --limit 2
```

Esse modo usa as respostas gold como saida do "modelo" e serve apenas para validar IO, metricas e summaries.

## Configuracoes

- `configs/eval.yaml`: paths, Spider dev, MMLU 150, geracao deterministica e SQLite timeout.
- `configs/train_lora_exp_a.yaml`: LoRA conservador, LR `1e-4`, 1 epoca.
- `configs/train_lora_exp_b.yaml`: LoRA mais agressivo, LR `2e-4`, 2 epocas.
- `configs/train_qlora_t4_template.yaml`: fallback T4 com QLoRA 4-bit.

Todas as configs usam paths relativos.

## Artefatos esperados

Depois de cada benchmark:

```text
outputs/<exp>/
├── environment.json
├── spider_predictions.jsonl
├── spider_metrics.json
├── mmlu_predictions.jsonl
├── mmlu_metrics.json
├── benchmark_summary.json
└── summary.md
```

Depois de cada treino:

```text
outputs/<exp>/
├── adapter/
├── tokenizer/
├── training_config.yaml
├── train_logs.json
├── dataset_preview.json
└── environment.json
```

## Metrica Spider

`custom_metrics.ExecutionAccuracy` herda de `deepeval.metrics.BaseMetric` quando DeepEval esta instalado. A metrica:

- extrai SQL de markdown e texto extra;
- aceita apenas `SELECT` ou `WITH`;
- bloqueia comandos destrutivos;
- executa SQL prevista e gold no SQLite em modo read-only;
- compara resultados ignorando ordem quando nao ha `ORDER BY`;
- preserva ordem quando ha `ORDER BY`;
- retorna `1.0` ou `0.0` e registra `error_type`.

## Colab

Abra `notebooks/colab_runner.ipynb`. O notebook apenas clona/entra no repositorio, instala dependencias e executa os comandos acima. A logica fica nos modulos e scripts do repositorio.

## Observacoes operacionais

- Nao commite `data/raw`, `data/processed`, `outputs`, tokens ou `.env`.
- Use `HF_TOKEN` no ambiente se o Hugging Face Hub exigir autenticacao.
- A avaliacao final deve usar Spider dev completo e MMLU 150 completo.
- O treino real depende de GPU, VRAM, download do modelo e datasets completos.

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
src/                logica principal
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

No Colab, o `requirements.txt` nao fixa `torch`: ele usa a versao CUDA/PyTorch ja
fornecida pelo runtime para evitar conflito com `torchvision` e drivers. Se voce
instalou uma versao anterior deste arquivo e viu conflitos de `numpy`, `pandas`,
`protobuf` ou `torch`, reinicie o runtime antes de seguir.

Se o treino falhar com `Found an incompatible version of torchao`, atualize o
codigo e reinstale as dependencias:

```bash
git pull origin main
pip install -r requirements.txt
```

Como alternativa emergencial no Colab, remova o pacote incompatível:

```bash
pip uninstall -y torchao
```

Se o treino LoRA em L4 falhar com CUDA OOM, confirme que os configs estao na
versao atual. O perfil L4 usa microbatch 2, acumulacao 4, sequencia 2048 e
gradient checkpointing para manter batch efetivo 8 sem quantizacao 4-bit:

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_seq_length: 2048
completion_only_loss: true
eos_token: "<|im_end|>"
gradient_checkpointing: true
```

## Dados

O comando de preparo do Spider agora tem uma camada de ingestao. Se `data/raw/spider`
ja existir, ele usa essa pasta. Caso contrario, ele pode importar de um diretorio,
de um ZIP/TAR ou tentar baixar de um dataset Hugging Face configuravel.

Estrutura final esperada:

```text
data/raw/spider/
├── train_spider.json
├── dev.json
├── tables.json
└── database/<db_id>/<db_id>.sqlite
```

O MMLU e baixado via Hugging Face Datasets por `scripts.prepare_mmlu`.

## Comandos principais

Preparar Spider com aquisicao automatica:

```bash
python -m scripts.prepare_spider --data_dir data/raw/spider --output_dir data/processed/spider
```

Importar Spider de um ZIP ou pasta do Drive/local:

```bash
python -m scripts.prepare_spider \
  --data_dir data/raw/spider \
  --output_dir data/processed/spider \
  --source_path /content/drive/MyDrive/spider.zip
```

Forcar download via Hugging Face:

```bash
python -m scripts.prepare_spider \
  --data_dir data/raw/spider \
  --output_dir data/processed/spider \
  --source hf \
  --hf_repo dreamerdeo/multispider
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

Treinar Experimento C:

```bash
python -m scripts.train --config configs/train_lora_exp_c.yaml
```

Treinar Experimento D, diagnostico de LR mais alto:

```bash
python -m scripts.train --config configs/train_lora_exp_d.yaml
```

Ao reexecutar um experimento no mesmo runtime, remova ou mova o diretorio
`outputs/<exp>` antigo antes do treino para evitar misturar checkpoints e
metricas de configuracoes anteriores.

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

Avaliar Exp C:

```bash
python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_c
```

Avaliar Exp D:

```bash
python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/exp_d
```

Diagnostico Spider sem `stop_sequences` para investigar saidas vazias do Exp C:

```bash
python -m scripts.evaluate_spider \
  --config configs/eval_spider_nostop.yaml \
  --model_path outputs/exp_c \
  --output_dir outputs/diagnostics/exp_c_spider_nostop
```

Avaliar o checkpoint de 1 epoca do Exp C:

```bash
python -m scripts.evaluate_spider \
  --config configs/eval_spider_nostop.yaml \
  --model_path outputs/exp_c/checkpoint-875 \
  --output_dir outputs/diagnostics/exp_c_ckpt875_spider_nostop
```

As avaliacoes usam `eval_batch_size` independente do treino. A configuracao padrao
usa batch 4 para Spider e batch 16 para MMLU em `configs/eval.yaml`; se a GPU
continuar com folga, esses valores podem ser aumentados, e se houver OOM basta
reduzi-los.

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

- `configs/eval.yaml`: paths, Spider dev, MMLU 150, geracao deterministica, batch de avaliacao e SQLite timeout.
- `configs/eval_spider_nostop.yaml`: diagnostico Spider sem `stop_sequences`, para separar erro real de EOS/stopping de corte prematuro da saida.
- `configs/train_lora_exp_a.yaml`: LoRA leve, `q_proj/v_proj`, LR `1e-4`, 1 epoca.
- `configs/train_lora_exp_b.yaml`: LoRA medio, `q_proj/k_proj/v_proj/o_proj`, LR `1e-4`, 1 epoca.
- `configs/train_lora_exp_c.yaml`: mesmo Exp B com 2 epocas para medir efeito da segunda epoca.
- `configs/train_lora_exp_d.yaml`: mesmo Exp B com LR `2e-4`, para testar se `1e-4` foi conservador.
- `configs/train_qlora_t4_template.yaml`: fallback T4 com QLoRA 4-bit.

Todas as configs usam paths relativos.

O dataset de treino e montado em formato `prompt` + `completion`, nao em um
campo unico `text`. Com `completion_only_loss: true`, o TRL calcula a loss
apenas nos tokens da SQL esperada. A completion recebe o EOS do Qwen
`<|im_end|>` para ensinar o modelo a parar depois da query. Antes de chamar o
`SFTTrainer`, o projeto tokeniza esses campos e cria `completion_mask`
explicitamente, evitando desalinhamento na fronteira `prompt`/`completion`.

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
- corta continuacoes de prompt como `Example`, `Schema`, `Question`, `SQL`,
  `Output: SQL`, `Task completed. SQL` e `Explanation`, inclusive quando a
  continuacao aparece na mesma linha;
- aceita apenas `SELECT` ou `WITH`;
- bloqueia comandos destrutivos;
- executa SQL prevista e gold no SQLite em modo read-only;
- le texto SQLite como bytes para evitar falhas de UTF-8 do banco serem
  confundidas com erro de SQL do modelo;
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

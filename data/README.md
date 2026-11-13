# Dados locais

Uso Spider train para fine-tuning e para os exemplos few-shot, Spider dev para avaliação e uma suíte MMLU de 150 questões como medida auxiliar de regressão. Os arquivos de runtime em `raw/` e `processed/` ficam fora do Git.

## Spider

O preparo aceita uma pasta já existente, um diretório/ZIP/TAR informado por `--source_path` ou uma fonte Hugging Face. A fonte automática configurada é `dreamerdeo/multispider`.

```text
data/raw/spider/
├── train_spider.json
├── dev.json
├── tables.json
└── database/<db_id>/<db_id>.sqlite
```

Na raiz do repositório:

```sh
python -m scripts.prepare_spider \
  --data_dir data/raw/spider --output_dir data/processed/spider
```

O comando gera `train.jsonl`, `dev.jsonl`, `schemas.json` e `metadata.json` em `data/processed/spider/`. Confira `missing_databases` e `unreadable_databases` no metadata antes de avaliar. A avaliação completa precisa dos 1034 exemplos dev e dos respectivos bancos SQLite.

Para importar um acervo local, acrescente `--source_path /caminho/spider.zip`. A [reprodução](../docs/reproducao.md) apresenta as alternativas e os cuidados com reexecuções.

## Suíte MMLU persistida

Preservei a suíte final em [`reports/evaluation/mmlu_150_suite.json`](../reports/evaluation/mmlu_150_suite.json). Para repetir a avaliação com as mesmas perguntas e exemplos:

```sh
mkdir -p data/processed
cp reports/evaluation/mmlu_150_suite.json data/processed/mmlu_150_suite.json
```

O JSON contém `metadata`, `questions` e `few_shot_examples`. São 50 questões de `college_computer_science`, 50 de `philosophy` e 50 de `econometrics`, com cinco exemplos few-shot por subcategoria. A suíte não representa o MMLU completo.

Para gerar uma nova suíte a partir de `cais/mmlu`, use `python -m scripts.prepare_mmlu --config configs/eval.yaml`. Esse comando escreve no caminho indicado em `paths.mmlu_suite_path`; preserve a suíte final antes de regenerá-la. Use `--mock` apenas com uma configuração que aponte para outro diretório. Perguntas sintéticas e respostas gold de smoke tests não são resultados do modelo.

Os datasets mantêm suas próprias condições de uso e atribuições; consulte [publicação](../docs/publicacao.md). Não inclua tokens ou dados privados nos arquivos versionados.

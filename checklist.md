# Checklist de progresso - TP2 NLP/Text-to-SQL

## Base do repositorio

- [x] Estrutura de pastas criada conforme PRD.
- [x] Logica principal em `src/tp2/`.
- [x] Scripts CLI em `scripts/`.
- [x] Configuracoes YAML em `configs/`.
- [x] `requirements.txt` criado com versoes fixadas.
- [x] `.gitignore` configurado para dados, outputs, caches e segredos.
- [x] `README.md` com instalacao, dados, treino, avaliacao e smoke tests.
- [x] `Makefile` com atalhos principais.
- [x] Notebook Colab runner sem logica exclusiva.

## Dados

- [x] `prepare_spider` implementado.
- [x] Ingestao automatica do Spider por pasta, arquivo compactado ou Hugging Face.
- [x] Conversao de Spider train/dev para JSONL implementada.
- [x] Serializacao de schemas Spider implementada.
- [x] Validacao leve de bancos SQLite Spider implementada.
- [x] `prepare_mmlu` implementado.
- [x] Selecao deterministica MMLU por seed implementada.
- [x] Suite MMLU com 50 STEM, 50 Humanidades e 50 Sociais configurada.
- [x] Modo mock de MMLU para smoke test implementado.

## Prompts e inferencia

- [x] Prompt Spider com schema, few-shot e instrucao de SQL-only.
- [x] Prompt MMLU 5-shot com alternativas A/B/C/D.
- [x] Geracao deterministica configurada (`temperature=0`, `do_sample=false`).
- [x] Extracao robusta de SQL de markdown/texto extra.
- [x] Parsing de resposta MMLU A/B/C/D.

## Metrica Spider

- [x] `ExecutionAccuracy` em `custom_metrics/execution_accuracy.py`.
- [x] Heranca de `deepeval.metrics.BaseMetric` quando DeepEval esta instalado.
- [x] Execucao SQLite read-only.
- [x] Bloqueio de comandos SQL destrutivos.
- [x] Comparacao ignora ordem sem `ORDER BY`.
- [x] Comparacao preserva ordem com `ORDER BY`.
- [x] Erros retornam `0.0` com `error_type`.
- [x] Testes unitarios da metrica.

## Treinamento

- [x] `python -m scripts.train --config ...` implementado.
- [x] Seeds fixadas antes do treino.
- [x] Formatacao SFT a partir do Spider train.
- [x] Carregamento de tokenizer/modelo implementado.
- [x] LoRA implementado via PEFT.
- [x] QLoRA/T4 template implementado.
- [x] Salvamento de adapter, tokenizer, config, ambiente e logs implementado.
- [ ] Treino real Exp A executado no Colab/GPU.
- [ ] Treino real Exp B executado no Colab/GPU.

## Avaliacao

- [x] `evaluate_spider` implementado.
- [x] `evaluate_mmlu` implementado.
- [x] `run_benchmarks` implementado.
- [x] Predicoes JSONL e metricas JSON implementadas.
- [x] Calculo de deltas contra baseline implementado.
- [x] `summary.md` por experimento implementado.
- [x] Modo mock para smoke tests implementado.
- [ ] Avaliacao real do baseline no Spider dev completo.
- [ ] Avaliacao real do baseline no MMLU 150.
- [ ] Avaliacao real Exp A no Spider/MMLU.
- [ ] Avaliacao real Exp B no Spider/MMLU.

## Testes

- [x] Testes de `sql_utils`.
- [x] Testes de `ExecutionAccuracy`.
- [x] Testes de prompt.
- [x] Testes de parsing MMLU.
- [x] Smoke test com Spider/MMLU mock e SQLite temporario.
- [x] `python -m pytest` passou localmente no `.venv` leve.
- [x] `python -m compileall -q src custom_metrics scripts tests` passou.
- [x] CLI `prepare_mmlu --mock --limit_per_category 2` passou.

## Entrega final

- [x] README documenta comandos reais solicitados.
- [x] README documenta que treino/inferencia real dependem de GPU/modelo/datasets.
- [x] Dados e outputs grandes ficam fora do Git.
- [x] Repositorio preparado para push em `mh131105/TP2_NLP`.

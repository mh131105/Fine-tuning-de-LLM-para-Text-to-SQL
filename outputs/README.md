# Saídas de execução

Os comandos escrevem os arquivos de runtime em `outputs/`. O Git ignora essas saídas; preserve os arquivos importantes antes de encerrar uma sessão Colab ou repetir um experimento. As evidências selecionadas para leitura pública ficam em [`reports/evaluation/`](../reports/evaluation/), e o adapter escolhido está no [Hugging Face](https://huggingface.co/mh131105/qwen2.5-3b-text-to-sql-lora-exp-f).

As pastas dos experimentos finais são `base` e `exp_a` até `exp_h`. O relatório recomenda `exp_f`; `exp_g` teve o maior resultado Spider. A configuração `exp_i` é exploratória e fica fora da comparação final A–H.

## Após o treinamento

```text
outputs/<exp>/
├── adapter/                 adapter_config.json e pesos LoRA
├── tokenizer/               tokenizer salvo no treino
├── checkpoint-*/            checkpoints por época, conforme a configuração
├── training_config.yaml     cópia da configuração utilizada
├── train_logs.json          perda, duração e métricas do treinamento
├── dataset_preview.json     prévia dos exemplos formatados
└── environment.json         ambiente registrado
```

O adapter depende do modelo-base `Qwen/Qwen2.5-3B-Instruct`. Não trate a configuração YAML, o notebook ou os logs como substitutos dos pesos reais.

## Após o benchmark

```text
outputs/<modelo>/
├── spider_predictions.jsonl
├── spider_metrics.json
├── mmlu_predictions.jsonl
├── mmlu_metrics.json
├── benchmark_summary.json
├── summary.md
└── environment.json
```

`run_benchmarks` executa Spider e MMLU e calcula diferenças em relação a `outputs/base/`, conforme `configs/eval.yaml`. Rode o baseline antes dos adapters. Os avaliadores sobrescrevem seus arquivos de saída; use outra pasta ou preserve a execução anterior antes de repetir.

Confira a presença de `adapter/adapter_config.json` e dos pesos antes de avaliar um experimento treinado. O carregador atual usa o modelo-base se não encontra um adapter no caminho informado. Uma pasta `outputs/exp_f/` vazia, portanto, não representa F.

Resultados com `--mock` usam respostas gold como saída e servem para verificar o encadeamento. Guarde-os separados das evidências reais. Veja os comandos e o checklist de preservação em [reprodução](../docs/reproducao.md).

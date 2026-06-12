[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2IQ_15wY)
# 2º Trabalho Prático — Text-to-SQL via Fine-Tuning

**ICC220 / PPGINF528 — 2026/01** · Profs. André Carvalho e Altigran da Silva
Análise quantitativa do trade-off entre especialização (Text-to-SQL) e generalização (MMLU) em LLMs via fine-tuning com LoRA/QLoRA.

> **Este é o repositório-modelo (starter) do trabalho.** Mantenha a estrutura de pastas
> e os nomes de arquivo. Preencha os stubs, complete este README com as instruções reais
> de execução e adicione o `RELATORIO.pdf` na raiz. Não apague pastas da estrutura.

---

## Regras de entrega (leia antes de começar)

A entrega é o **conteúdo versionado neste repositório do GitHub Classroom**. Vale nota de organização e a forma é levada a sério.

- ❌ **Proibido** commitar `.zip`/`.rar`/`.tar`, a `venv/`, `__pycache__/`, checkpoints de modelo ou os datasets (Spider/MMLU). O `.gitignore` já cobre isso — não o remova.
- ✅ A estrutura (`scripts/`, `custom_metrics/`, `tests/`, `configs/`, `requirements.txt`, `README.md`) fica **na raiz**. Não aninhe tudo dentro de uma subpasta.
- ✅ O relatório é **um único PDF**, chamado **`RELATORIO.pdf`**, na **raiz**, com **texto selecionável** (não escaneado/imagem) e que **abre** em qualquer leitor. Recomenda-se manter também a fonte (`.tex`/`.md`).
- ✅ `requirements.txt` com **versões fixadas** (`pacote==versao`), refletindo o que você realmente usou.
- ✅ Nomes de arquivo apenas com `[a-z0-9_-]`: sem espaços, acentos ou maiúsculas acentuadas.
- ✅ Preencha o `CHECKLIST.md` indicando onde cada requisito foi cumprido.

---

## Estrutura do repositório

```
.
├── README.md                       # este arquivo (complete com instruções reais)
├── RELATORIO.pdf                   # você adiciona (PDF, máx. 10 págs, IEEE/ACM)
├── requirements.txt                # versões FIXADAS
├── .gitignore
├── CHECKLIST.md                    # aderência aos requisitos
├── scripts/
│   ├── preprocess_spider.py        # Spider (train) -> formato de chat
│   ├── build_mmlu_suite.py         # monta as 150 questões (50 STEM / 50 Hum / 50 CS)
│   ├── train.py                    # fine-tuning LoRA/QLoRA
│   ├── eval_spider.py              # baseline + fine-tuned no dev split (Execution Accuracy)
│   └── eval_mmlu.py                # MMLU 5-shot, base e fine-tuned
├── custom_metrics/
│   └── execution_accuracy.py       # métrica DeepEval (VOCÊ implementa)
├── tests/
│   └── test_execution_accuracy.py  # avaliação via pytest
├── configs/
│   ├── lora_config_a.yaml          # configuração 1 de hiperparâmetros
│   └── lora_config_b.yaml          # configuração 2 (mínimo 2)
├── data/                           # datasets baixados aqui (NÃO versionar)
└── results/
    ├── baseline/                   # saídas/numeros do modelo base
    ├── finetuned/                  # saídas/numeros dos modelos fine-tuned
    └── mmlu/                       # acurácias MMLU (agregada e por categoria)
```

## Configuração do ambiente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Baixe os datasets para `data/` (eles não vão para o Git):

```bash
# Spider: site oficial -> data/spider/
# MMLU: Hugging Face Hub (cais/mmlu) -> baixado pelo build_mmlu_suite.py
python scripts/build_mmlu_suite.py
```

## Reprodução dos resultados (preencha com seus comandos reais)

```bash
# 1. Pré-processar o Spider train para formato de chat
python scripts/preprocess_spider.py

# 2. Baseline (modelo base, sem treino) no Spider dev
python scripts/eval_spider.py --model <BASE_MODEL> --split dev --out results/baseline/

# 3. Fine-tuning (rode para CADA config de hiperparâmetros)
python scripts/train.py --config configs/lora_config_a.yaml
python scripts/train.py --config configs/lora_config_b.yaml

# 4. Avaliar fine-tuned no Spider dev (MESMO procedimento do baseline)
python scripts/eval_spider.py --model <FT_MODEL_A> --split dev --out results/finetuned/

# 5. Métrica oficial via pytest
pytest tests/

# 6. Regressão MMLU (base e fine-tuned), 5-shot
python scripts/eval_mmlu.py --model <BASE_MODEL>  --out results/mmlu/
python scripts/eval_mmlu.py --model <FT_MODEL_A>  --out results/mmlu/
```

## Reprodutibilidade (obrigatório)

- **Seeds fixadas** em toda operação estocástica (init de pesos, amostragem, splits).
- **Decodificação determinística** na avaliação: greedy, `temperature = 0`.
- Documente o **hardware** (GPU/VRAM) no relatório.
- Discuta a possibilidade de **contaminação de dados** (Spider/MMLU vistos no pré-treino).

## Modelo e hardware

- Base sugerido (Colab T4, tier gratuito): `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` ou `microsoft/Phi-3.5-mini-instruct`.
- Documente o **checkpoint exato** usado.

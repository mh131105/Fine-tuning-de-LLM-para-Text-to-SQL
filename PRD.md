## 1. Resumo executivo

A solução proposta é um pipeline modular em Google Colab para medir o trade-off entre **especialização em Text-to-SQL** e **regressão de capacidade geral** após fine-tuning de um LLM. O PDF exige fine-tuning em Spider, avaliação por **Execution Accuracy customizada em DeepEval**, baseline no Spider dev, avaliação dos modelos fine-tuned com o mesmo procedimento, e análise de regressão em MMLU com 150 questões 5-shot. 

O pipeline usa `Qwen/Qwen2.5-Coder-3B` como modelo-alvo, com carregamento em 4 bits quando necessário. O fine-tuning será feito com **QLoRA** por padrão, pois o ambiente-alvo é Colab com GPU T4 ou similar. A arquitetura separa preparação de dados, prompts, geração, métrica, treinamento, avaliação, agregação e exportação de artefatos.

A métrica de Text-to-SQL será **Execution Accuracy**. As demais métricas de SQL serão diagnósticas, como erro de extração, erro SQLite, timeout e latência. Isso preserva o requisito do PDF de usar uma única métrica de avaliação da tarefa Text-to-SQL.

---

## 2. Premissas adotadas

* **Modelo:** usar `Qwen/Qwen2.5-Coder-3B`, conforme pedido. A família Qwen2.5-Coder é voltada para código e possui variante 3B disponível no Hugging Face. ([Hugging Face][1])
* **Observação sobre requisito instruct/chat:** o PDF pede um modelo open-source instruct/chat de 3-4B. O modelo indicado pelo usuário não traz `Instruct` no nome. Portanto, o pipeline deve registrar essa decisão em `outputs/metrics/model_metadata.json`. Se o professor exigir estritamente instruct/chat, trocar para `Qwen/Qwen2.5-Coder-3B-Instruct` sem mudar a arquitetura.
* **GPU:** Colab T4 é o alvo mínimo. L4/A100 reduzem tempo e permitem batch maior.
* **VRAM:** usar QLoRA 4-bit, batch 1, gradient accumulation e `max_seq_length=2048`.
* **Dataset Spider:** usar `train` apenas para fine-tuning e exemplos few-shot; usar `dev` apenas para avaliação. Spider é um benchmark Text-to-SQL com bancos SQLite e avaliação por execução. ([yale-lily.github.io][2])
* **Dataset MMLU:** usar Hugging Face `cais/mmlu`, com 150 questões, 3 categorias, 50 por categoria. O MMLU cobre múltiplas áreas como humanidades, ciências sociais e hard sciences. ([Hugging Face][3])
* **DeepEval:** implementar métrica customizada herdando `BaseMetric` e usando `LLMTestCase`. A documentação oficial orienta implementar `measure()` para calcular o score. ([DeepEval][4])
* **Treinamento:** usar Hugging Face TRL + PEFT + bitsandbytes. Bitsandbytes reduz uso de memória via quantização. ([Hugging Face][5])
* **Reprodutibilidade:** fixar seeds, geração greedy, hashes de datasets, hashes de prompts, hashes de configs e `requirements.lock.txt`.
* **Limitação:** a Execution Accuracy proposta deve seguir a semântica do Spider, mas não substitui integralmente o avaliador oficial se houver casos complexos de equivalência SQL.

---

## 3. Matriz de rastreabilidade

| ID     | Requisito do PDF                                       | Implementação proposta                                              | Arquivo/script responsável             | Métrica/evidência gerada                  | Status esperado |
| ------ | ------------------------------------------------------ | ------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------- | --------------- |
| RQ-001 | Avaliar fine-tuning em LLMs para Text-to-SQL           | Pipeline Colab com baseline, QLoRA e avaliação                      | `README.md`, `notebooks/*.ipynb`       | `outputs/metrics/experiment_summary.json` | Obrigatório     |
| RQ-002 | Medir ganho na tarefa-alvo                             | Comparar Spider dev baseline vs fine-tuned                          | `scripts/aggregate_results.py`         | `comparison_table.csv`                    | Obrigatório     |
| RQ-003 | Medir regressão de capacidade geral                    | Avaliar MMLU antes e depois do fine-tuning                          | `scripts/evaluate_mmlu.py`             | `mmlu_regression_table.md`                | Obrigatório     |
| RQ-004 | Usar Spider training split para fine-tuning            | Preparar JSONL somente com `train_spider.json`                      | `scripts/prepare_spider.py`            | `spider_train.jsonl`, hash                | Obrigatório     |
| RQ-005 | Não usar Spider dev no treinamento                     | Separação por config e validação de split                           | `configs/data.yaml`                    | `split_audit.json`                        | Obrigatório     |
| RQ-006 | Incluir schema no prompt                               | Serializar tabelas, colunas, PKs e FKs                              | `src/spider_schema.py`                 | `schema_serialization_samples.jsonl`      | Obrigatório     |
| RQ-007 | Implementar Execution Accuracy customizada             | Classe DeepEval `ExecutionAccuracyMetric`                           | `custom_metrics/execution_accuracy.py` | score `0.0` ou `1.0` por exemplo          | Obrigatório     |
| RQ-008 | Herdar `deepeval.metrics.BaseMetric`                   | Implementar classe compatível                                       | `custom_metrics/execution_accuracy.py` | teste pytest de herança                   | Obrigatório     |
| RQ-009 | Implementar `measure(self, test_case)`                 | Método recebe `LLMTestCase`                                         | `execution_accuracy.py`                | `test_execution_accuracy.py`              | Obrigatório     |
| RQ-010 | Extrair SQL da saída bruta                             | Remover markdown, prefixos e explicações                            | `src/sql_utils.py`                     | `sql_extraction_failure_rate`             | Obrigatório     |
| RQ-011 | Executar SQL predita em SQLite                         | Abrir banco correto em modo leitura                                 | `src/sql_utils.py`                     | `sqlite_execution_error_rate`             | Obrigatório     |
| RQ-012 | Executar SQL gold                                      | Usar `expected_output` do `LLMTestCase`                             | `execution_accuracy.py`                | logs por exemplo                          | Obrigatório     |
| RQ-013 | Comparar resultados com regra de ordem                 | Respeitar `ORDER BY`; ignorar ordem nos demais casos                | `execution_accuracy.py`                | `result_mismatch`                         | Obrigatório     |
| RQ-014 | Usar Execution Accuracy como única métrica Text-to-SQL | Usar EA como métrica de qualidade; demais métricas são diagnósticas | `evaluate_spider.py`                   | `execution_accuracy`                      | Obrigatório     |
| RQ-015 | Prompt few-shot com 3 exemplos do Spider train         | Fixar IDs dos 3 exemplos                                            | `configs/eval_spider.yaml`             | `fewshot_examples_hash`                   | Obrigatório     |
| RQ-016 | Baseline no Spider dev                                 | Rodar modelo base sem fine-tuning                                   | `scripts/run_baseline_spider.py`       | `outputs/baseline/spider_metrics.json`    | Obrigatório     |
| RQ-017 | Registrar SQL gerada e sucesso/falha                   | Exportar JSONL por exemplo                                          | `evaluate_spider.py`                   | `spider_predictions.jsonl`                | Obrigatório     |
| RQ-018 | Fine-tuning com LoRA ou QLoRA                          | Usar PEFT com QLoRA por padrão                                      | `scripts/train_qlora.py`               | adapters salvos                           | Obrigatório     |
| RQ-019 | Documentar rank, alpha, dropout e target modules       | Config YAML e metadados do treinamento                              | `configs/train_exp_*.yaml`             | `training_config_resolved.json`           | Obrigatório     |
| RQ-020 | Testar duas configurações de hiperparâmetros           | Experimento A e B                                                   | `train_exp_a.yaml`, `train_exp_b.yaml` | métricas por experimento                  | Obrigatório     |
| RQ-021 | Documentar hardware                                    | Detectar GPU, VRAM, CUDA e libs                                     | `src/environment.py`                   | `environment.json`                        | Obrigatório     |
| RQ-022 | Avaliar fine-tuned igual ao baseline                   | Reusar `evaluate_spider.py` e mesmo prompt                          | `scripts/evaluate_spider.py`           | comparação justa                          | Obrigatório     |
| RQ-023 | Integrar métrica a pytest                              | Testes unitários e integração                                       | `tests/test_execution_accuracy.py`     | relatório pytest                          | Obrigatório     |
| RQ-024 | MMLU com exatamente 150 questões                       | 50 STEM, 50 Humanidades, 50 Ciências Sociais                        | `scripts/prepare_mmlu.py`              | `mmlu_suite.jsonl`                        | Obrigatório     |
| RQ-025 | MMLU 5-shot                                            | Mesmo contexto 5-shot para todos os modelos                         | `src/prompts.py`                       | `mmlu_prompt_hash`                        | Obrigatório     |
| RQ-026 | Calcular acurácia MMLU                                 | Extrair A/B/C/D e comparar                                          | `evaluate_mmlu.py`                     | `mmlu_accuracy_*`                         | Obrigatório     |
| RQ-027 | Calcular variação percentual                           | Delta base vs fine-tuned                                            | `aggregate_results.py`                 | `mmlu_delta_*_percent`                    | Obrigatório     |
| RQ-028 | Repositório profissional                               | Estrutura com scripts, métricas, README e configs                   | repositório completo                   | árvore versionada                         | Obrigatório     |
| RQ-029 | `requirements.txt` com versões fixadas                 | Pin de dependências e lock gerado                                   | `requirements.txt`                     | `requirements_lock_hash`                  | Obrigatório     |
| RQ-030 | README de reprodução                                   | Passo a passo Colab                                                 | `README.md`                            | checklist executável                      | Obrigatório     |
| RQ-031 | Relatório técnico até 10 páginas                       | Gerar tabelas e resumo Markdown                                     | `docs/`, `outputs/report_assets/`      | assets prontos                            | Obrigatório     |
| RQ-032 | Fixar seeds                                            | Seed global Python, NumPy, Torch e datasets                         | `src/reproducibility.py`               | `seed` nos logs                           | Obrigatório     |
| RQ-033 | Usar geração determinística                            | `do_sample=false`, `temperature=0`, greedy                          | `configs/eval_*.yaml`                  | `generation_config.json`                  | Obrigatório     |
| RQ-034 | Discutir contaminação de dados                         | Seção no relatório e README                                         | `docs/experiment_plan.md`              | `contamination_discussion.md`             | Obrigatório     |

---

## 4. Arquitetura proposta

### Componentes

| Componente                             | Responsabilidade                                                   |
| -------------------------------------- | ------------------------------------------------------------------ |
| `src/config.py`                        | Carregar YAML, validar campos obrigatórios e gerar hash da config. |
| `src/environment.py`                   | Detectar GPU, VRAM, CUDA, Python, Torch e versões de libs.         |
| `src/reproducibility.py`               | Fixar seeds e registrar flags determinísticas.                     |
| `src/spider_schema.py`                 | Ler `tables.json` e bancos SQLite; serializar schemas.             |
| `src/prompts.py`                       | Construir prompt Text-to-SQL e prompt MMLU 5-shot.                 |
| `src/model_loader.py`                  | Carregar Qwen com quantização e adapters PEFT.                     |
| `src/generation.py`                    | Executar geração determinística e registrar tokens/latência.       |
| `src/sql_utils.py`                     | Extrair SQL, validar, executar SQLite e normalizar resultados.     |
| `custom_metrics/execution_accuracy.py` | Métrica DeepEval para Execution Accuracy.                          |
| `scripts/train_qlora.py`               | Treinar adapters QLoRA.                                            |
| `scripts/evaluate_spider.py`           | Avaliar baseline e fine-tuned no Spider dev.                       |
| `scripts/evaluate_mmlu.py`             | Avaliar MMLU 5-shot.                                               |
| `scripts/aggregate_results.py`         | Comparar baseline, exp A, exp B e gerar deltas.                    |
| `scripts/export_report_tables.py`      | Exportar tabelas Markdown/CSV para relatório.                      |

### Fluxo de dados

```text
PDF Spec
   ↓
Requirements Extraction
   ↓
Traceability Matrix
   ↓
Data Preparation
   ├── Spider train → Fine-tuning JSONL
   ├── Spider dev → Evaluation Cases
   └── MMLU subset → Generalization Suite
   ↓
Baseline Evaluation
   ↓
QLoRA Fine-tuning Experiments
   ├── Experiment A
   └── Experiment B
   ↓
Fine-tuned Spider Evaluation
   ↓
MMLU Regression Evaluation
   ↓
Metrics Aggregation
   ↓
Report Assets + README + User Stories
```

### Artefatos gerados

* Dados processados: `data/processed/*.jsonl`
* Logs: `outputs/metrics/*.jsonl`
* Predições Spider: `outputs/*/spider_predictions.jsonl`
* Métricas Spider: `outputs/*/spider_metrics.json`
* Métricas MMLU: `outputs/mmlu/*.json`
* Tabelas: `outputs/report_assets/*.md`, `outputs/metrics/*.csv`
* Checklist: `docs/reproducibility_checklist.md`

---

## 5. Estrutura de repositório

```text
project-root/
├── README.md
├── requirements.txt
├── requirements.lock.txt
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── model.yaml
│   ├── train_exp_a.yaml
│   ├── train_exp_b.yaml
│   ├── eval_spider.yaml
│   └── eval_mmlu.yaml
├── notebooks/
│   ├── 00_setup_colab.ipynb
│   ├── 01_prepare_data.ipynb
│   ├── 02_baseline_eval.ipynb
│   ├── 03_finetune_qlora.ipynb
│   ├── 04_eval_finetuned.ipynb
│   └── 05_analysis_report.ipynb
├── scripts/
│   ├── prepare_spider.py
│   ├── prepare_mmlu.py
│   ├── run_baseline_spider.py
│   ├── train_qlora.py
│   ├── evaluate_spider.py
│   ├── evaluate_mmlu.py
│   ├── aggregate_results.py
│   └── export_report_tables.py
├── custom_metrics/
│   ├── __init__.py
│   └── execution_accuracy.py
├── src/
│   ├── config.py
│   ├── environment.py
│   ├── prompts.py
│   ├── spider_schema.py
│   ├── sql_utils.py
│   ├── model_loader.py
│   ├── generation.py
│   ├── metrics_io.py
│   └── reproducibility.py
├── tests/
│   ├── test_sql_extraction.py
│   ├── test_execution_accuracy.py
│   ├── test_schema_serialization.py
│   └── test_reproducibility.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── outputs/
│   ├── baseline/
│   ├── finetuned_exp_a/
│   ├── finetuned_exp_b/
│   ├── mmlu/
│   ├── metrics/
│   └── report_assets/
└── docs/
    ├── user_stories.md
    ├── architecture.md
    ├── traceability_matrix.md
    ├── experiment_plan.md
    └── reproducibility_checklist.md
```

### Função das pastas

* `configs/`: todos os parâmetros editáveis.
* `notebooks/`: execução guiada no Colab.
* `scripts/`: execução por linha de comando.
* `custom_metrics/`: métrica DeepEval exigida.
* `src/`: código reutilizável.
* `tests/`: testes automatizados.
* `data/raw/`: Spider e MMLU originais.
* `data/processed/`: JSONL prontos para treino e avaliação.
* `outputs/`: resultados, logs, adapters e tabelas.
* `docs/`: documentação técnica e checklist de reprodução.

---

## 6. Pipeline ponta a ponta

1. **Setup**

   * Clonar repositório.
   * Instalar dependências.
   * Registrar ambiente em `outputs/metrics/environment.json`.
   * Fixar seed global.

2. **Download/preparação dos dados**

   * Baixar ou montar Spider em `data/raw/spider/`.
   * Validar existência de `train_spider.json`, `dev.json`, `tables.json` e `database/`.
   * Preparar MMLU com exatamente 150 questões.

3. **Construção dos prompts**

   * Serializar schema SQLite.
   * Fixar 3 exemplos Spider train para few-shot.
   * Gerar prompt com schema do banco-alvo, pergunta e instrução “retorne apenas SQL”.

4. **Baseline Spider**

   * Carregar modelo base.
   * Gerar SQL com decodificação determinística.
   * Avaliar com Execution Accuracy.
   * Exportar predições e métricas.

5. **Fine-tuning experimento A**

   * Carregar modelo em 4 bits.
   * Aplicar LoRA com configuração A.
   * Salvar adapters em `outputs/finetuned_exp_a/adapters/`.

6. **Fine-tuning experimento B**

   * Repetir treino com outra taxa de aprendizado ou número de épocas.
   * Salvar adapters em `outputs/finetuned_exp_b/adapters/`.

7. **Avaliação Spider fine-tuned**

   * Carregar base + adapter.
   * Reusar o mesmo prompt e a mesma métrica.
   * Exportar resultados por experimento.

8. **Avaliação MMLU**

   * Avaliar modelo base e fine-tuned com os mesmos 5 exemplos de contexto.
   * Extrair resposta A/B/C/D.
   * Calcular acurácia por categoria e agregada.

9. **Agregação de métricas**

   * Consolidar baseline, exp A e exp B.
   * Calcular deltas Text-to-SQL e MMLU.

10. **Exportação para relatório**

* Gerar tabelas Markdown, CSV e JSON.
* Gerar resumo de análise de erros.

---

## 7. User Stories

### EPIC-01 — Ambiente, portabilidade e reprodutibilidade

### US-001 — Configurar ambiente Colab

**Épico:** EPIC-01
**Como:** engenheiro de ML
**Quero:** configurar o ambiente Colab
**Para:** executar o pipeline sem ajustes manuais

**Contexto técnico:**
Notebook `00_setup_colab.ipynb` prepara dependências, diretórios e acesso ao projeto.

**Requisitos funcionais:**

* Criar diretórios padrão.
* Instalar dependências.
* Verificar GPU disponível.

**Requisitos não funcionais:**

* Rodar em T4.
* Não exigir paths locais fixos.

**Critérios de aceitação:**

* Dado que o Colab iniciou.
* Quando a célula de setup rodar.
* Então `outputs/metrics/environment.json` deve existir.

**Evidências esperadas:**

* `environment.json`.
* Log de instalação.
* Versão Python registrada.

**Arquivos impactados:**

* `notebooks/00_setup_colab.ipynb`
* `src/environment.py`

**Métricas geradas:**

* `python_version`
* `cuda_version`
* `gpu_name`

**Dependências:**

* Nenhuma.

**Riscos:**

* Colab sem GPU.

**Definition of Done:**

* Setup executa do zero em runtime limpo.

---

### US-002 — Fixar dependências

**Épico:** EPIC-01
**Como:** avaliador do projeto
**Quero:** dependências com versões fixadas
**Para:** reproduzir o ambiente experimental

**Contexto técnico:**
O PDF exige `requirements.txt` com versões fixadas.

**Requisitos funcionais:**

* Criar `requirements.txt`.
* Gerar `requirements.lock.txt` com `pip freeze`.
* Calcular hash do lock.

**Requisitos não funcionais:**

* Evitar dependências sem versão.
* Registrar conflitos de instalação.

**Critérios de aceitação:**

* Dado que o setup terminou.
* Quando `pip freeze` for executado.
* Então `requirements_lock_hash` deve ser exportado.

**Evidências esperadas:**

* `requirements.txt`.
* `requirements.lock.txt`.
* `outputs/metrics/reproducibility.json`.

**Arquivos impactados:**

* `requirements.txt`
* `README.md`

**Métricas geradas:**

* `requirements_lock_hash`

**Dependências:**

* US-001.

**Riscos:**

* Incompatibilidade entre `torch`, `bitsandbytes` e CUDA.

**Definition of Done:**

* `pip install -r requirements.txt` conclui sem erro no Colab.

---

### US-003 — Detectar e registrar hardware

**Épico:** EPIC-01
**Como:** autor do relatório
**Quero:** registrar GPU e VRAM
**Para:** documentar o hardware usado no fine-tuning

**Contexto técnico:**
O PDF exige documentar modelo de GPU e VRAM.

**Requisitos funcionais:**

* Detectar GPU via Torch.
* Registrar memória total e alocada.
* Registrar CUDA.

**Requisitos não funcionais:**

* Funcionar também sem GPU.
* Não interromper setup se CUDA ausente.

**Critérios de aceitação:**

* Dado que há GPU.
* Quando `environment.py` rodar.
* Então `gpu_name` e `gpu_vram` devem aparecer no JSON.

**Evidências esperadas:**

* `outputs/metrics/environment.json`.

**Arquivos impactados:**

* `src/environment.py`

**Métricas geradas:**

* `gpu_name`
* `gpu_vram`
* `cuda_version`

**Dependências:**

* US-001.

**Riscos:**

* API CUDA indisponível.

**Definition of Done:**

* Ambiente registra hardware antes de qualquer treino.

---

### US-004 — Fixar seeds

**Épico:** EPIC-01
**Como:** pesquisador
**Quero:** fixar todas as seeds
**Para:** permitir reprodução independente

**Contexto técnico:**
Seeds devem cobrir Python, NumPy, Torch e amostragem de datasets.

**Requisitos funcionais:**

* Implementar `set_global_seed(seed)`.
* Registrar seed em todos os outputs.
* Usar seed na seleção MMLU.

**Requisitos não funcionais:**

* Não depender de ordem instável de arquivos.

**Critérios de aceitação:**

* Dado que `seed=42`.
* Quando o pipeline rodar duas vezes.
* Então hashes de prompts e datasets devem ser iguais.

**Evidências esperadas:**

* `reproducibility.json`.
* `dataset_hash`.

**Arquivos impactados:**

* `src/reproducibility.py`

**Métricas geradas:**

* `seed`

**Dependências:**

* US-001.

**Riscos:**

* Operações CUDA não totalmente determinísticas.

**Definition of Done:**

* Teste `test_reproducibility.py` passa.

---

### US-005 — Centralizar configurações

**Épico:** EPIC-01
**Como:** engenheiro
**Quero:** editar parâmetros em YAML
**Para:** trocar hiperparâmetros sem alterar código

**Contexto técnico:**
Configs centralizam modelo, dados, treino, avaliação e paths.

**Requisitos funcionais:**

* Criar `configs/*.yaml`.
* Validar campos obrigatórios.
* Exportar config resolvida.

**Requisitos não funcionais:**

* Falhar com erro claro se campo estiver ausente.

**Critérios de aceitação:**

* Dado um YAML válido.
* Quando o script iniciar.
* Então a config resolvida deve ser salva em `outputs/metrics/`.

**Evidências esperadas:**

* `resolved_config.json`.
* `config_hash`.

**Arquivos impactados:**

* `src/config.py`
* `configs/*.yaml`

**Métricas geradas:**

* `config_hash`

**Dependências:**

* US-002.

**Riscos:**

* Divergência entre notebook e script.

**Definition of Done:**

* Todos os scripts recebem `--config`.

---

### EPIC-02 — Preparação dos dados Spider

### US-006 — Preparar Spider train

**Épico:** EPIC-02
**Como:** engenheiro de dados
**Quero:** processar o Spider train
**Para:** gerar exemplos de fine-tuning sem vazamento do dev

**Contexto técnico:**
Somente `train_spider.json` pode alimentar treino e few-shot.

**Requisitos funcionais:**

* Ler `train_spider.json`.
* Anexar schema serializado.
* Gerar exemplos instruction/chat.

**Requisitos não funcionais:**

* Preservar `db_id`, pergunta e SQL gold.

**Critérios de aceitação:**

* Dado o Spider bruto.
* Quando `prepare_spider.py --split train` rodar.
* Então `spider_train.jsonl` deve ser criado.

**Evidências esperadas:**

* `data/processed/spider_train.jsonl`.
* `spider_train_hash`.

**Arquivos impactados:**

* `scripts/prepare_spider.py`

**Métricas geradas:**

* `total_train_examples`
* `dataset_hash`

**Dependências:**

* US-008.

**Riscos:**

* Estrutura local do Spider diferente.

**Definition of Done:**

* Nenhum exemplo dev aparece no JSONL de treino.

---

### US-007 — Preparar Spider dev

**Épico:** EPIC-02
**Como:** avaliador
**Quero:** preparar Spider dev
**Para:** avaliar baseline e fine-tuned com o mesmo conjunto

**Contexto técnico:**
O dev não pode ser usado em treino.

**Requisitos funcionais:**

* Ler `dev.json`.
* Gerar casos de avaliação.
* Registrar hash do split.

**Requisitos não funcionais:**

* Não gerar campo usado por trainer.

**Critérios de aceitação:**

* Dado o Spider dev.
* Quando `prepare_spider.py --split dev` rodar.
* Então `spider_dev_eval.jsonl` deve existir.

**Evidências esperadas:**

* `data/processed/spider_dev_eval.jsonl`.
* `split_audit.json`.

**Arquivos impactados:**

* `scripts/prepare_spider.py`

**Métricas geradas:**

* `total_dev_examples`
* `dev_hash`

**Dependências:**

* US-008.

**Riscos:**

* Uso acidental do dev no treino.

**Definition of Done:**

* Validador bloqueia `dev` em `train_qlora.py`.

---

### US-008 — Serializar schemas SQLite

**Épico:** EPIC-02
**Como:** modelo de geração
**Quero:** receber o schema do banco
**Para:** gerar tabelas e colunas corretas

**Contexto técnico:**
O PDF exige schema linking no prompt.

**Requisitos funcionais:**

* Ler `tables.json`.
* Incluir tabelas, colunas, PKs e FKs.
* Gerar formato textual compacto.

**Requisitos não funcionais:**

* Limitar tamanho para T4.
* Manter ordem determinística.

**Critérios de aceitação:**

* Dado um `db_id`.
* Quando a serialização rodar.
* Então o schema deve conter tabelas e colunas do banco correto.

**Evidências esperadas:**

* `schema_serialization_samples.jsonl`.

**Arquivos impactados:**

* `src/spider_schema.py`

**Métricas geradas:**

* `average_schema_tokens`

**Dependências:**

* Spider bruto.

**Riscos:**

* Prompt exceder contexto.

**Definition of Done:**

* `test_schema_serialization.py` passa.

---

### US-009 — Gerar dataset de treino JSONL

**Épico:** EPIC-02
**Como:** trainer
**Quero:** consumir JSONL padronizado
**Para:** treinar com TRL sem lógica extra

**Contexto técnico:**
Cada linha contém prompt e resposta SQL.

**Requisitos funcionais:**

* Criar campo `messages` ou `text`.
* Incluir SQL gold como resposta.
* Registrar formato usado.

**Requisitos não funcionais:**

* Compatível com tokenizer Qwen.
* Não incluir exemplos dev.

**Critérios de aceitação:**

* Dado o train processado.
* Quando o trainer carregar o JSONL.
* Então cada exemplo deve ter entrada e saída válidas.

**Evidências esperadas:**

* `spider_train_sft.jsonl`.

**Arquivos impactados:**

* `scripts/prepare_spider.py`
* `src/prompts.py`

**Métricas geradas:**

* `average_train_prompt_tokens`

**Dependências:**

* US-006.

**Riscos:**

* Chat template ausente no modelo base.

**Definition of Done:**

* Smoke test treina 2 batches sem erro.

---

### US-010 — Validar integridade dos exemplos

**Épico:** EPIC-02
**Como:** avaliador
**Quero:** validar exemplos processados
**Para:** evitar erro silencioso de dados

**Contexto técnico:**
Validação verifica SQL, banco, schema e split.

**Requisitos funcionais:**

* Verificar `db_id`.
* Verificar existência do SQLite.
* Verificar SQL gold não vazio.
* Calcular hashes.

**Requisitos não funcionais:**

* Falhar com relatório claro.

**Critérios de aceitação:**

* Dado JSONL processado.
* Quando o validador rodar.
* Então inconsistências devem ir para `data_validation_report.json`.

**Evidências esperadas:**

* `data_validation_report.json`.

**Arquivos impactados:**

* `scripts/prepare_spider.py`

**Métricas geradas:**

* `invalid_examples`
* `dataset_hash`

**Dependências:**

* US-006, US-007.

**Riscos:**

* SQL gold com dialeto não aceito por SQLite.

**Definition of Done:**

* Pipeline para se houver exemplo inválido crítico.

---

### EPIC-03 — Prompt Engineering para baseline

### US-011 — Criar template few-shot

**Épico:** EPIC-03
**Como:** engenheiro de prompts
**Quero:** um template Text-to-SQL fixo
**Para:** comparar baseline e fine-tuned de forma justa

**Contexto técnico:**
Prompt deve conter instrução, 3 exemplos, schema e pergunta.

**Requisitos funcionais:**

* Criar `prompts/text_to_sql_fewshot_v1.txt`.
* Instruir “retorne apenas SQL”.
* Incluir placeholders.

**Requisitos não funcionais:**

* Determinístico.
* Versionado por hash.

**Critérios de aceitação:**

* Dado o template.
* Quando for renderizado.
* Então deve conter schema, pergunta e 3 exemplos.

**Evidências esperadas:**

* `prompt_template_hash`.

**Arquivos impactados:**

* `src/prompts.py`
* `configs/eval_spider.yaml`

**Métricas geradas:**

* `prompt_template_hash`

**Dependências:**

* US-008.

**Riscos:**

* Prompt longo demais.

**Definition of Done:**

* Template é reutilizado por baseline e fine-tuned.

---

### US-012 — Selecionar 3 exemplos fixos

**Épico:** EPIC-03
**Como:** pesquisador
**Quero:** fixar 3 exemplos do Spider train
**Para:** evitar variação entre avaliações

**Contexto técnico:**
Os IDs devem ficar em config.

**Requisitos funcionais:**

* Selecionar exemplos representativos.
* Registrar IDs.
* Impedir uso de exemplos dev.

**Requisitos não funcionais:**

* Seleção estável.
* Hash exportado.

**Critérios de aceitação:**

* Dado `fewshot_example_ids`.
* Quando o prompt for renderizado.
* Então os mesmos 3 exemplos devem aparecer.

**Evidências esperadas:**

* `fewshot_examples.json`.
* `fewshot_examples_hash`.

**Arquivos impactados:**

* `configs/eval_spider.yaml`

**Métricas geradas:**

* `fewshot_examples_hash`

**Dependências:**

* US-006.

**Riscos:**

* Exemplo escolhido com SQL muito longa.

**Definition of Done:**

* IDs fixos documentados no README.

---

### US-013 — Garantir schema linking no prompt

**Épico:** EPIC-03
**Como:** modelo
**Quero:** receber o schema do banco-alvo
**Para:** gerar SQL executável

**Contexto técnico:**
O schema do banco da pergunta deve ser incluído em cada prompt.

**Requisitos funcionais:**

* Injetar schema do `db_id`.
* Incluir pergunta do exemplo.
* Não misturar schemas de outros bancos.

**Requisitos não funcionais:**

* Sanitizar quebras excessivas.

**Critérios de aceitação:**

* Dado um exemplo dev.
* Quando o prompt for gerado.
* Então o schema deve corresponder ao `db_id`.

**Evidências esperadas:**

* `prompt_render_samples.jsonl`.

**Arquivos impactados:**

* `src/prompts.py`
* `src/spider_schema.py`

**Métricas geradas:**

* `average_prompt_tokens`

**Dependências:**

* US-008.

**Riscos:**

* Schema truncado afetar acurácia.

**Definition of Done:**

* Amostra de 10 prompts é exportada.

---

### US-014 — Versionar prompt de avaliação

**Épico:** EPIC-03
**Como:** avaliador
**Quero:** versionar o prompt
**Para:** auditar mudanças de resultado

**Contexto técnico:**
Mudanças no prompt alteram o baseline.

**Requisitos funcionais:**

* Calcular SHA256 do template.
* Salvar prompt renderizado de amostra.
* Registrar versão em métricas.

**Requisitos não funcionais:**

* Não sobrescrever versão anterior sem novo hash.

**Critérios de aceitação:**

* Dado um template.
* Quando a avaliação rodar.
* Então o hash deve aparecer nos resultados.

**Evidências esperadas:**

* `prompt_template_hash`.
* `prompt_samples.md`.

**Arquivos impactados:**

* `src/metrics_io.py`

**Métricas geradas:**

* `prompt_template_hash`

**Dependências:**

* US-011.

**Riscos:**

* Comparar experimentos com prompts diferentes.

**Definition of Done:**

* Agregador bloqueia comparação se hashes divergirem.

---

### EPIC-04 — Métrica customizada Execution Accuracy

### US-015 — Implementar extração robusta de SQL

**Épico:** EPIC-04
**Como:** avaliador
**Quero:** extrair SQL da saída bruta
**Para:** avaliar modelos mesmo com ruído textual

**Contexto técnico:**
A saída pode conter markdown, explicação ou prefixos.

**Requisitos funcionais:**

* Remover blocos ```sql.
* Remover prefixos como `SQL:`.
* Capturar primeira query plausível.
* Retornar erro se não houver SQL.

**Requisitos não funcionais:**

* Não modificar SQL válida desnecessariamente.

**Critérios de aceitação:**

* Dado uma saída com markdown.
* Quando `extract_sql()` rodar.
* Então apenas a SQL deve retornar.

**Evidências esperadas:**

* `test_sql_extraction.py`.

**Arquivos impactados:**

* `src/sql_utils.py`

**Métricas geradas:**

* `sql_extraction_failure_rate`

**Dependências:**

* Nenhuma.

**Riscos:**

* Cortar subqueries complexas.

**Definition of Done:**

* Testes cobrem markdown, texto antes/depois e ausência de SQL.

---

### US-016 — Implementar conexão SQLite segura

**Épico:** EPIC-04
**Como:** avaliador
**Quero:** executar consultas em modo seguro
**Para:** evitar escrita ou travamento do banco

**Contexto técnico:**
A conexão deve ser read-only e com timeout.

**Requisitos funcionais:**

* Abrir SQLite com `mode=ro`.
* Usar `PRAGMA query_only=ON`.
* Aplicar timeout por query.
* Capturar erro de banco ausente.

**Requisitos não funcionais:**

* Não alterar arquivos `.sqlite`.

**Critérios de aceitação:**

* Dado uma query `DROP TABLE`.
* Quando a execução ocorrer.
* Então deve falhar sem modificar o banco.

**Evidências esperadas:**

* `SQL_EXECUTION_ERROR`.
* Teste unitário.

**Arquivos impactados:**

* `src/sql_utils.py`

**Métricas geradas:**

* `sqlite_execution_error_rate`
* `timeout_rate`

**Dependências:**

* Spider SQLite.

**Riscos:**

* Query muito lenta.

**Definition of Done:**

* Timeout e modo read-only testados.

---

### US-017 — Executar SQL predita e SQL gold

**Épico:** EPIC-04
**Como:** métrica DeepEval
**Quero:** executar SQL predita e gold
**Para:** comparar resultados reais

**Contexto técnico:**
`actual_output` contém a saída do modelo e `expected_output` contém SQL gold.

**Requisitos funcionais:**

* Executar predita.
* Executar gold.
* Registrar resultados ou erro.
* Preservar `db_id`.

**Requisitos não funcionais:**

* Tempo de execução registrado por exemplo.

**Critérios de aceitação:**

* Dado `LLMTestCase` válido.
* Quando `measure()` rodar.
* Então deve retornar `1.0` ou `0.0`.

**Evidências esperadas:**

* `spider_eval_results.jsonl`.

**Arquivos impactados:**

* `custom_metrics/execution_accuracy.py`

**Métricas geradas:**

* `execution_accuracy`
* `latency_seconds`

**Dependências:**

* US-016.

**Riscos:**

* `db_path` não encontrado no contexto.

**Definition of Done:**

* Caso correto retorna `1.0`; caso errado retorna `0.0`.

---

### US-018 — Comparar resultados corretamente

**Épico:** EPIC-04
**Como:** avaliador
**Quero:** comparar resultados com regra de ordem
**Para:** aproximar a semântica do Spider

**Contexto técnico:**
Sem `ORDER BY`, a ordem das linhas não deve importar.

**Requisitos funcionais:**

* Detectar `ORDER BY`.
* Usar lista ordenada quando houver `ORDER BY`.
* Usar multiset quando não houver.
* Normalizar tipos simples.

**Requisitos não funcionais:**

* Preservar duplicatas.

**Critérios de aceitação:**

* Dado resultados iguais em ordem diferente.
* Quando não houver `ORDER BY`.
* Então o score deve ser `1.0`.

**Evidências esperadas:**

* `test_execution_accuracy.py`.

**Arquivos impactados:**

* `execution_accuracy.py`
* `sql_utils.py`

**Métricas geradas:**

* `result_mismatch_rate`

**Dependências:**

* US-017.

**Riscos:**

* Diferenças numéricas de float.

**Definition of Done:**

* Testes cobrem ordem, duplicatas e mismatch.

---

### US-019 — Integrar métrica ao DeepEval

**Épico:** EPIC-04
**Como:** avaliador
**Quero:** usar DeepEval com métrica customizada
**Para:** atender ao framework exigido no PDF

**Contexto técnico:**
Classe herda `BaseMetric` e implementa `measure()`.

**Requisitos funcionais:**

* Criar `ExecutionAccuracyMetric`.
* Implementar `is_successful()`.
* Expor `score` e `reason`.

**Requisitos não funcionais:**

* Compatível com `pytest`.

**Critérios de aceitação:**

* Dado um `LLMTestCase`.
* Quando DeepEval chamar a métrica.
* Então o score deve ser calculado sem LLM judge.

**Evidências esperadas:**

* Teste pytest com DeepEval.

**Arquivos impactados:**

* `custom_metrics/execution_accuracy.py`

**Métricas geradas:**

* `execution_accuracy`

**Dependências:**

* DeepEval fixado no requirements.

**Riscos:**

* Mudança de API entre versões do DeepEval.

**Definition of Done:**

* Testes passam com versão fixada.

---

### US-020 — Criar testes pytest da métrica

**Épico:** EPIC-04
**Como:** mantenedor
**Quero:** testes automatizados da métrica
**Para:** evitar regressões na avaliação

**Contexto técnico:**
Testes usam banco SQLite mínimo em `data/samples/`.

**Requisitos funcionais:**

* Testar SQL correta.
* Testar SQL errada.
* Testar erro de sintaxe.
* Testar `ORDER BY`.

**Requisitos não funcionais:**

* Rodar em CPU em menos de 30 segundos.

**Critérios de aceitação:**

* Dado `pytest`.
* Quando os testes rodarem.
* Então todos devem passar sem GPU.

**Evidências esperadas:**

* Relatório pytest.

**Arquivos impactados:**

* `tests/test_execution_accuracy.py`

**Métricas geradas:**

* `pytest_passed`

**Dependências:**

* US-015 a US-019.

**Riscos:**

* Testes dependerem do Spider completo.

**Definition of Done:**

* Testes usam banco sintético pequeno.

---

### EPIC-05 — Avaliação do modelo base no Spider dev

### US-021 — Carregar modelo base

**Épico:** EPIC-05
**Como:** engenheiro de ML
**Quero:** carregar o modelo base
**Para:** executar o baseline sem fine-tuning

**Contexto técnico:**
Carregamento usa `AutoModelForCausalLM` e `AutoTokenizer`.

**Requisitos funcionais:**

* Carregar `Qwen/Qwen2.5-Coder-3B`.
* Configurar `pad_token=eos_token` se necessário.
* Registrar tokenizer e chat template.

**Requisitos não funcionais:**

* Suportar 4-bit para inferência.

**Critérios de aceitação:**

* Dado modelo configurado.
* Quando `run_baseline_spider.py` rodar.
* Então o modelo deve gerar uma saída.

**Evidências esperadas:**

* `model_metadata.json`.

**Arquivos impactados:**

* `src/model_loader.py`

**Métricas geradas:**

* `model_name`
* `model_revision`

**Dependências:**

* US-001.

**Riscos:**

* Modelo base sem comportamento instruct forte.

**Definition of Done:**

* Smoke test gera SQL para 1 exemplo.

---

### US-022 — Executar inferência determinística

**Épico:** EPIC-05
**Como:** pesquisador
**Quero:** geração greedy
**Para:** reproduzir as saídas de avaliação

**Contexto técnico:**
O PDF exige temperatura 0 e decodificação determinística.

**Requisitos funcionais:**

* Usar `do_sample=false`.
* Usar `temperature=0.0`.
* Usar `num_beams=1`.
* Registrar config de geração.

**Requisitos não funcionais:**

* Mesmo input deve gerar mesmo output.

**Critérios de aceitação:**

* Dado o mesmo prompt.
* Quando a geração rodar duas vezes.
* Então a saída deve ser idêntica.

**Evidências esperadas:**

* `generation_config.json`.

**Arquivos impactados:**

* `src/generation.py`

**Métricas geradas:**

* `average_generation_tokens`
* `average_latency_seconds`

**Dependências:**

* US-021.

**Riscos:**

* Kernels CUDA não determinísticos em casos raros.

**Definition of Done:**

* Teste de repetição passa em amostra pequena.

---

### US-023 — Avaliar Spider dev

**Épico:** EPIC-05
**Como:** avaliador
**Quero:** avaliar o dev split inteiro
**Para:** obter baseline quantitativo

**Contexto técnico:**
Executa prompts dev e chama Execution Accuracy.

**Requisitos funcionais:**

* Iterar sobre `spider_dev_eval.jsonl`.
* Gerar SQL.
* Avaliar cada exemplo.
* Registrar progresso.

**Requisitos não funcionais:**

* Permitir `--limit` para smoke test.

**Critérios de aceitação:**

* Dado o dev processado.
* Quando a avaliação terminar.
* Então métricas agregadas devem existir.

**Evidências esperadas:**

* `outputs/baseline/spider_metrics.json`.

**Arquivos impactados:**

* `scripts/run_baseline_spider.py`
* `scripts/evaluate_spider.py`

**Métricas geradas:**

* `execution_accuracy`
* `total_examples`

**Dependências:**

* US-007, US-019.

**Riscos:**

* Tempo alto no Colab.

**Definition of Done:**

* Baseline roda com `--limit 10` e com split completo.

---

### US-024 — Exportar resultados por exemplo

**Épico:** EPIC-05
**Como:** analista
**Quero:** JSONL com cada predição
**Para:** fazer análise de erros sem rerodar o modelo

**Contexto técnico:**
Cada linha contém pergunta, gold, raw output, SQL extraída e erro.

**Requisitos funcionais:**

* Salvar `predicted_raw`.
* Salvar `predicted_sql`.
* Salvar `error_type`.
* Salvar latência.

**Requisitos não funcionais:**

* Formato JSONL válido.

**Critérios de aceitação:**

* Dado uma avaliação.
* Quando cada exemplo finalizar.
* Então uma linha deve ser escrita no JSONL.

**Evidências esperadas:**

* `spider_predictions.jsonl`.

**Arquivos impactados:**

* `src/metrics_io.py`

**Métricas geradas:**

* `failed_examples`
* `invalid_sql_rate`

**Dependências:**

* US-023.

**Riscos:**

* Perda de resultados se runtime cair.

**Definition of Done:**

* Escrita incremental ativada.

---

### US-025 — Calcular baseline agregado

**Épico:** EPIC-05
**Como:** autor do relatório
**Quero:** métricas agregadas do baseline
**Para:** comparar com fine-tuned

**Contexto técnico:**
Agregação calcula Execution Accuracy e diagnósticos.

**Requisitos funcionais:**

* Contar corretos e falhos.
* Calcular taxa de erro SQL.
* Exportar JSON e Markdown.

**Requisitos não funcionais:**

* Não recalcular geração.

**Critérios de aceitação:**

* Dado `spider_predictions.jsonl`.
* Quando agregador rodar.
* Então `spider_metrics.json` deve ser criado.

**Evidências esperadas:**

* `spider_metrics.json`.
* `baseline_summary.md`.

**Arquivos impactados:**

* `scripts/aggregate_results.py`

**Métricas geradas:**

* `execution_accuracy`
* `correct_examples`
* `failed_examples`

**Dependências:**

* US-024.

**Riscos:**

* JSONL truncado.

**Definition of Done:**

* Agregador valida quantidade de linhas esperada.

---

### EPIC-06 — Fine-tuning com LoRA/QLoRA

### US-026 — Configurar QLoRA

**Épico:** EPIC-06
**Como:** engenheiro de ML
**Quero:** treinar em 4 bits
**Para:** caber na VRAM da T4

**Contexto técnico:**
Usar bitsandbytes com NF4 e double quant.

**Requisitos funcionais:**

* Configurar `load_in_4bit`.
* Usar `bnb_4bit_quant_type=nf4`.
* Preparar modelo para k-bit training.

**Requisitos não funcionais:**

* Reduzir uso de VRAM.

**Critérios de aceitação:**

* Dado GPU T4.
* Quando o treino iniciar.
* Então o modelo deve carregar sem OOM.

**Evidências esperadas:**

* `training_environment.json`.
* `peak_gpu_memory_mb`.

**Arquivos impactados:**

* `scripts/train_qlora.py`
* `src/model_loader.py`

**Métricas geradas:**

* `quantization_mode`
* `peak_gpu_memory_mb`

**Dependências:**

* US-021.

**Riscos:**

* Bitsandbytes incompatível com CUDA.

**Definition of Done:**

* Smoke test de treino roda 2 batches.

---

### US-027 — Definir configuração LoRA editável

**Épico:** EPIC-06
**Como:** pesquisador
**Quero:** editar rank, alpha, dropout e módulos
**Para:** documentar e variar o fine-tuning

**Contexto técnico:**
LoRA fica em YAML.

**Requisitos funcionais:**

* Definir `r`, `alpha`, `dropout`.
* Validar `target_modules`.
* Logar módulos encontrados e ausentes.

**Requisitos não funcionais:**

* Não falhar sem diagnóstico.

**Critérios de aceitação:**

* Dado target module inexistente.
* Quando o modelo carregar.
* Então o log deve indicar módulo ausente.

**Evidências esperadas:**

* `lora_modules_report.json`.

**Arquivos impactados:**

* `configs/train_exp_*.yaml`
* `src/model_loader.py`

**Métricas geradas:**

* `lora_r`
* `lora_alpha`
* `lora_dropout`

**Dependências:**

* US-026.

**Riscos:**

* Nomes de módulos variam por arquitetura.

**Definition of Done:**

* Relatório lista módulos aplicados.

---

### US-028 — Executar experimento A

**Épico:** EPIC-06
**Como:** pesquisador
**Quero:** treinar configuração A
**Para:** medir uma condição experimental

**Contexto técnico:**
Experimento A usa `lr=2e-4`, 1 época.

**Requisitos funcionais:**

* Ler `train_exp_a.yaml`.
* Treinar com Spider train.
* Salvar adapter.

**Requisitos não funcionais:**

* Checkpoints por época.

**Critérios de aceitação:**

* Dado config A.
* Quando o treino terminar.
* Então adapter e métricas devem existir.

**Evidências esperadas:**

* `outputs/finetuned_exp_a/adapters/`.
* `train_metrics.json`.

**Arquivos impactados:**

* `scripts/train_qlora.py`

**Métricas geradas:**

* `train_loss`
* `training_runtime_seconds`

**Dependências:**

* US-009, US-026.

**Riscos:**

* Treino demorar mais que sessão Colab.

**Definition of Done:**

* Adapter A pode ser recarregado para inferência.

---

### US-029 — Executar experimento B

**Épico:** EPIC-06
**Como:** pesquisador
**Quero:** treinar configuração B
**Para:** comparar hiperparâmetros

**Contexto técnico:**
Experimento B usa `lr=1e-4`, 2 épocas.

**Requisitos funcionais:**

* Ler `train_exp_b.yaml`.
* Treinar no mesmo dataset.
* Salvar adapter separado.

**Requisitos não funcionais:**

* Não sobrescrever experimento A.

**Critérios de aceitação:**

* Dado config B.
* Quando o treino terminar.
* Então adapter B deve ficar em pasta própria.

**Evidências esperadas:**

* `outputs/finetuned_exp_b/adapters/`.

**Arquivos impactados:**

* `configs/train_exp_b.yaml`
* `scripts/train_qlora.py`

**Métricas geradas:**

* `train_loss`
* `num_train_epochs`

**Dependências:**

* US-028.

**Riscos:**

* Overfitting maior por mais épocas.

**Definition of Done:**

* Adapter B pode ser avaliado no mesmo script.

---

### US-030 — Salvar adapters e metadados

**Épico:** EPIC-06
**Como:** mantenedor
**Quero:** salvar adapters e configs resolvidas
**Para:** reproduzir cada modelo fine-tuned

**Contexto técnico:**
Cada experimento deve ser autocontido.

**Requisitos funcionais:**

* Salvar adapter PEFT.
* Salvar tokenizer quando necessário.
* Salvar config resolvida.
* Salvar hash do dataset.

**Requisitos não funcionais:**

* Não salvar pesos completos se não necessário.

**Critérios de aceitação:**

* Dado treino finalizado.
* Quando pasta do experimento for aberta.
* Então adapter, config e metadados devem existir.

**Evidências esperadas:**

* `adapter_config.json`.
* `training_config_resolved.json`.

**Arquivos impactados:**

* `scripts/train_qlora.py`

**Métricas geradas:**

* `dataset_hash`
* `config_hash`

**Dependências:**

* US-028, US-029.

**Riscos:**

* Adapter incompatível com revisão diferente do modelo base.

**Definition of Done:**

* Metadata inclui `base_model_name` e `model_revision`.

---

### US-031 — Registrar métricas de treinamento

**Épico:** EPIC-06
**Como:** autor do relatório
**Quero:** logs de loss, tempo e memória
**Para:** documentar custo experimental

**Contexto técnico:**
Logs devem alimentar tabelas do relatório.

**Requisitos funcionais:**

* Registrar loss por step.
* Registrar runtime.
* Registrar pico de GPU.
* Registrar tokens/s quando possível.

**Requisitos não funcionais:**

* Exportar JSONL e CSV.

**Critérios de aceitação:**

* Dado um treino.
* Quando cada logging step ocorrer.
* Então uma linha deve ser escrita no log.

**Evidências esperadas:**

* `training_log.jsonl`.
* `training_metrics.csv`.

**Arquivos impactados:**

* `src/metrics_io.py`
* `scripts/train_qlora.py`

**Métricas geradas:**

* `train_loss`
* `tokens_per_second`
* `peak_gpu_memory_mb`

**Dependências:**

* US-028.

**Riscos:**

* Métrica tokens/s indisponível em alguns trainers.

**Definition of Done:**

* Relatório consegue montar tabela de hiperparâmetros e custo.

---

### EPIC-07 — Avaliação dos modelos fine-tuned

### US-032 — Carregar modelo fine-tuned

**Épico:** EPIC-07
**Como:** avaliador
**Quero:** carregar base + adapter
**Para:** avaliar cada experimento

**Contexto técnico:**
Modelo base é carregado e adapter PEFT aplicado.

**Requisitos funcionais:**

* Receber path do adapter.
* Validar compatibilidade.
* Registrar adapter usado.

**Requisitos não funcionais:**

* Suportar inferência 4-bit.

**Critérios de aceitação:**

* Dado adapter salvo.
* Quando avaliação iniciar.
* Então modelo fine-tuned deve gerar saída.

**Evidências esperadas:**

* `finetuned_model_metadata.json`.

**Arquivos impactados:**

* `src/model_loader.py`

**Métricas geradas:**

* `adapter_path`
* `base_model_name`

**Dependências:**

* US-030.

**Riscos:**

* Adapter salvo incompleto.

**Definition of Done:**

* Smoke test gera SQL com adapter A e B.

---

### US-033 — Avaliar modelo fine-tuned no Spider dev

**Épico:** EPIC-07
**Como:** pesquisador
**Quero:** avaliar fine-tuned no mesmo dev
**Para:** medir ganho de especialização

**Contexto técnico:**
Reusa `evaluate_spider.py`.

**Requisitos funcionais:**

* Avaliar exp A.
* Avaliar exp B.
* Usar mesmo prompt e métrica do baseline.

**Requisitos não funcionais:**

* Bloquear comparação se prompt hash divergir.

**Critérios de aceitação:**

* Dado adapter A ou B.
* Quando avaliação rodar.
* Então JSONL e métricas devem ser salvos.

**Evidências esperadas:**

* `outputs/finetuned_exp_a/spider_metrics.json`.
* `outputs/finetuned_exp_b/spider_metrics.json`.

**Arquivos impactados:**

* `scripts/evaluate_spider.py`

**Métricas geradas:**

* `execution_accuracy`
* `invalid_sql_rate`

**Dependências:**

* US-032.

**Riscos:**

* Fine-tuning piorar geração SQL.

**Definition of Done:**

* Exp A e B avaliados com mesmo procedimento.

---

### US-034 — Comparar com baseline

**Épico:** EPIC-07
**Como:** autor do relatório
**Quero:** comparar baseline e fine-tuned
**Para:** quantificar ganho ou perda em Text-to-SQL

**Contexto técnico:**
Comparação usa apenas Execution Accuracy como métrica principal.

**Requisitos funcionais:**

* Carregar métricas dos três modelos.
* Calcular delta absoluto.
* Gerar tabela comparativa.

**Requisitos não funcionais:**

* Validar mesmo dataset hash.

**Critérios de aceitação:**

* Dado métricas de baseline, A e B.
* Quando agregador rodar.
* Então tabela comparativa deve ser gerada.

**Evidências esperadas:**

* `comparison_table.csv`.
* `spider_results_table.md`.

**Arquivos impactados:**

* `scripts/aggregate_results.py`

**Métricas geradas:**

* `spider_delta_absolute`

**Dependências:**

* US-025, US-033.

**Riscos:**

* Comparar execuções com datasets diferentes.

**Definition of Done:**

* Agregador valida hashes antes de comparar.

---

### US-035 — Exportar tabela de resultados

**Épico:** EPIC-07
**Como:** equipe de relatório
**Quero:** tabelas em CSV e Markdown
**Para:** colar no relatório sem retrabalho

**Contexto técnico:**
Exporta resumo por modelo.

**Requisitos funcionais:**

* Exportar CSV.
* Exportar Markdown.
* Incluir baseline, A e B.

**Requisitos não funcionais:**

* Cabe em relatório de 10 páginas.

**Critérios de aceitação:**

* Dado métricas consolidadas.
* Quando exportação rodar.
* Então arquivos de tabela devem existir.

**Evidências esperadas:**

* `outputs/report_assets/spider_results_table.md`.

**Arquivos impactados:**

* `scripts/export_report_tables.py`

**Métricas geradas:**

* Métricas já agregadas.

**Dependências:**

* US-034.

**Riscos:**

* Tabela extensa demais.

**Definition of Done:**

* Tabela inclui modelo, EA, erros SQL e latência média.

---

### EPIC-08 — Avaliação MMLU e regressão de capacidade

### US-036 — Construir suíte MMLU com 150 questões

**Épico:** EPIC-08
**Como:** pesquisador
**Quero:** montar uma suíte MMLU fixa
**Para:** medir generalização com tamanho controlado

**Contexto técnico:**
50 questões por categoria, uma subcategoria por grupo.

**Requisitos funcionais:**

* Selecionar 50 STEM.
* Selecionar 50 Humanidades.
* Selecionar 50 Ciências Sociais.
* Salvar IDs e hash.

**Requisitos não funcionais:**

* Amostragem determinística por seed.

**Critérios de aceitação:**

* Dado `eval_mmlu.yaml`.
* Quando `prepare_mmlu.py` rodar.
* Então `mmlu_suite.jsonl` deve ter exatamente 150 linhas.

**Evidências esperadas:**

* `mmlu_suite.jsonl`.
* `mmlu_suite_hash`.

**Arquivos impactados:**

* `scripts/prepare_mmlu.py`

**Métricas geradas:**

* `mmlu_total_questions`

**Dependências:**

* US-004.

**Riscos:**

* Nome da subcategoria não existir no dataset.

**Definition of Done:**

* Validador falha se total for diferente de 150.

---

### US-037 — Criar prompt 5-shot fixo

**Épico:** EPIC-08
**Como:** avaliador
**Quero:** usar o mesmo prompt 5-shot
**Para:** comparar modelos de forma justa

**Contexto técnico:**
Usa 5 exemplos fixos por subcategoria.

**Requisitos funcionais:**

* Selecionar 5 exemplos de contexto.
* Reusar para todos os modelos.
* Instruir resposta A/B/C/D.

**Requisitos não funcionais:**

* Registrar hash do prompt.

**Critérios de aceitação:**

* Dado uma questão MMLU.
* Quando o prompt for renderizado.
* Então os mesmos 5 exemplos devem aparecer.

**Evidências esperadas:**

* `mmlu_fewshot_examples.json`.
* `mmlu_prompt_hash`.

**Arquivos impactados:**

* `src/prompts.py`

**Métricas geradas:**

* `mmlu_prompt_hash`

**Dependências:**

* US-036.

**Riscos:**

* Exemplo 5-shot vazar resposta de teste.

**Definition of Done:**

* Exemplos 5-shot e questões avaliadas são conjuntos distintos.

---

### US-038 — Avaliar modelo base no MMLU

**Épico:** EPIC-08
**Como:** pesquisador
**Quero:** medir MMLU do modelo base
**Para:** obter referência de generalização

**Contexto técnico:**
Avaliação determinística, resposta final A/B/C/D.

**Requisitos funcionais:**

* Gerar resposta para 150 questões.
* Extrair alternativa.
* Calcular acurácia por categoria.

**Requisitos não funcionais:**

* Registrar erros de parsing.

**Critérios de aceitação:**

* Dado modelo base carregado.
* Quando MMLU rodar.
* Então métricas base devem ser exportadas.

**Evidências esperadas:**

* `outputs/mmlu/mmlu_baseline.json`.

**Arquivos impactados:**

* `scripts/evaluate_mmlu.py`

**Métricas geradas:**

* `mmlu_accuracy_overall`
* `mmlu_accuracy_stem`

**Dependências:**

* US-021, US-037.

**Riscos:**

* Modelo responde com texto em vez de alternativa.

**Definition of Done:**

* Parser extrai ou registra `PARSE_ERROR`.

---

### US-039 — Avaliar modelos fine-tuned no MMLU

**Épico:** EPIC-08
**Como:** pesquisador
**Quero:** medir MMLU dos modelos fine-tuned
**Para:** identificar regressão de capacidade

**Contexto técnico:**
Mesmo prompt 5-shot usado no baseline.

**Requisitos funcionais:**

* Avaliar exp A.
* Avaliar exp B.
* Salvar respostas por questão.

**Requisitos não funcionais:**

* Validar mesmo suite hash.

**Critérios de aceitação:**

* Dado adapter fine-tuned.
* Quando MMLU rodar.
* Então métricas do adapter devem ser salvas.

**Evidências esperadas:**

* `mmlu_exp_a.json`.
* `mmlu_exp_b.json`.

**Arquivos impactados:**

* `scripts/evaluate_mmlu.py`

**Métricas geradas:**

* `mmlu_accuracy_humanities`
* `mmlu_accuracy_social_sciences`

**Dependências:**

* US-032, US-038.

**Riscos:**

* Fine-tuning reduzir obediência ao formato A/B/C/D.

**Definition of Done:**

* Exp A e B avaliados com o mesmo `mmlu_suite_hash`.

---

### US-040 — Calcular regressão agregada e por categoria

**Épico:** EPIC-08
**Como:** autor do relatório
**Quero:** calcular deltas MMLU
**Para:** reportar perda ou ganho de generalização

**Contexto técnico:**
Delta absoluto e percentual por categoria e geral.

**Requisitos funcionais:**

* Comparar baseline vs A.
* Comparar baseline vs B.
* Exportar tabela de regressão.

**Requisitos não funcionais:**

* Não dividir por zero sem tratamento.

**Critérios de aceitação:**

* Dado métricas MMLU.
* Quando agregador rodar.
* Então deltas devem aparecer por categoria.

**Evidências esperadas:**

* `mmlu_regression_table.md`.

**Arquivos impactados:**

* `scripts/aggregate_results.py`

**Métricas geradas:**

* `mmlu_delta_overall_percent`
* `mmlu_delta_stem_percent`

**Dependências:**

* US-038, US-039.

**Riscos:**

* Amostra de 150 ter alta variância.

**Definition of Done:**

* Relatório inclui delta absoluto e percentual.

---

### EPIC-09 — Métricas, logs e análise automatizada

### US-041 — Criar logger estruturado

**Épico:** EPIC-09
**Como:** engenheiro
**Quero:** logs JSONL estruturados
**Para:** auditar cada etapa do pipeline

**Contexto técnico:**
Cada evento terá timestamp, etapa, config hash e payload.

**Requisitos funcionais:**

* Criar logger JSONL.
* Escrever eventos de setup, treino e avaliação.
* Incluir hashes.

**Requisitos não funcionais:**

* Escrita incremental.

**Critérios de aceitação:**

* Dado qualquer script.
* Quando uma etapa começar ou terminar.
* Então um evento deve ser registrado.

**Evidências esperadas:**

* `pipeline_events.jsonl`.

**Arquivos impactados:**

* `src/metrics_io.py`

**Métricas geradas:**

* Métricas de infraestrutura e execução.

**Dependências:**

* US-005.

**Riscos:**

* Logs grandes demais.

**Definition of Done:**

* Logs podem ser lidos por `aggregate_results.py`.

---

### US-042 — Exportar métricas em JSONL

**Épico:** EPIC-09
**Como:** analista
**Quero:** métricas por exemplo em JSONL
**Para:** filtrar falhas e gerar análise de erros

**Contexto técnico:**
Spider e MMLU salvam uma linha por item.

**Requisitos funcionais:**

* Exportar JSONL por exemplo.
* Validar schema dos campos.
* Salvar erro quando ocorrer.

**Requisitos não funcionais:**

* Não perder dados em queda de runtime.

**Critérios de aceitação:**

* Dado uma avaliação interrompida.
* Quando o arquivo for aberto.
* Então linhas já concluídas devem estar válidas.

**Evidências esperadas:**

* `spider_predictions.jsonl`.
* `mmlu_predictions.jsonl`.

**Arquivos impactados:**

* `src/metrics_io.py`

**Métricas geradas:**

* Métricas por exemplo.

**Dependências:**

* US-024, US-038.

**Riscos:**

* JSON inválido por escrita parcial.

**Definition of Done:**

* Escrita usa flush por linha.

---

### US-043 — Exportar tabelas CSV

**Épico:** EPIC-09
**Como:** equipe de análise
**Quero:** tabelas CSV
**Para:** analisar resultados em planilha

**Contexto técnico:**
CSV consolida modelos, métricas e deltas.

**Requisitos funcionais:**

* Exportar Spider CSV.
* Exportar MMLU CSV.
* Exportar treino CSV.

**Requisitos não funcionais:**

* Colunas estáveis entre execuções.

**Critérios de aceitação:**

* Dado resultados completos.
* Quando exportação rodar.
* Então CSVs devem abrir sem pós-processamento.

**Evidências esperadas:**

* `comparison_table.csv`.
* `training_summary.csv`.

**Arquivos impactados:**

* `scripts/export_report_tables.py`

**Métricas geradas:**

* Todas as agregadas.

**Dependências:**

* US-034, US-040.

**Riscos:**

* Valores faltantes.

**Definition of Done:**

* CSV usa `null` ou vazio padronizado para ausentes.

---

### US-044 — Gerar resumo Markdown para relatório

**Épico:** EPIC-09
**Como:** autor do relatório
**Quero:** tabelas Markdown prontas
**Para:** reduzir edição manual

**Contexto técnico:**
Gera tabelas compatíveis com relatório IEEE/ACM.

**Requisitos funcionais:**

* Criar tabela Spider.
* Criar tabela LoRA.
* Criar tabela MMLU.
* Criar resumo de ambiente.

**Requisitos não funcionais:**

* Tabelas curtas e legíveis.

**Critérios de aceitação:**

* Dado métricas agregadas.
* Quando exportação rodar.
* Então arquivos `.md` devem existir.

**Evidências esperadas:**

* `spider_results_table.md`.
* `mmlu_regression_table.md`.

**Arquivos impactados:**

* `scripts/export_report_tables.py`

**Métricas geradas:**

* Nenhuma nova.

**Dependências:**

* US-043.

**Riscos:**

* Relatório ultrapassar 10 páginas.

**Definition of Done:**

* Markdown pode ser colado diretamente no relatório.

---

### US-045 — Gerar análise automática de erros

**Épico:** EPIC-09
**Como:** pesquisador
**Quero:** exemplos de falhas selecionados automaticamente
**Para:** discutir 2-3 casos no relatório

**Contexto técnico:**
Seleciona falhas representativas por tipo de erro.

**Requisitos funcionais:**

* Agrupar por `error_type`.
* Selecionar exemplos curtos.
* Incluir pergunta, gold e predita.

**Requisitos não funcionais:**

* Não usar exemplos corretos como falha.

**Critérios de aceitação:**

* Dado predições Spider.
* Quando análise rodar.
* Então `error_analysis.md` deve conter 2-3 falhas.

**Evidências esperadas:**

* `outputs/report_assets/error_analysis.md`.

**Arquivos impactados:**

* `scripts/aggregate_results.py`

**Métricas geradas:**

* `error_type_counts`

**Dependências:**

* US-024.

**Riscos:**

* Falhas muito longas para relatório.

**Definition of Done:**

* Saída inclui motivo e SQL extraída.

---

### EPIC-10 — Estrutura profissional de repositório

### US-046 — Criar README

**Épico:** EPIC-11
**Como:** avaliador externo
**Quero:** README completo
**Para:** reproduzir os resultados

**Contexto técnico:**
README cobre instalação, dados, treino, avaliação e relatório.

**Requisitos funcionais:**

* Explicar setup Colab.
* Explicar comandos.
* Explicar métricas.
* Explicar reprodução.

**Requisitos não funcionais:**

* Comandos copiáveis.

**Critérios de aceitação:**

* Dado runtime limpo.
* Quando seguir o README.
* Então pipeline deve executar em ordem.

**Evidências esperadas:**

* `README.md`.

**Arquivos impactados:**

* `README.md`

**Métricas geradas:**

* Nenhuma.

**Dependências:**

* Todas as histórias principais.

**Riscos:**

* README desatualizado após mudança de script.

**Definition of Done:**

* README referencia configs e outputs reais.

---

### US-047 — Criar documentação de arquitetura

**Épico:** EPIC-11
**Como:** professor avaliador
**Quero:** arquitetura documentada
**Para:** entender responsabilidades do sistema

**Contexto técnico:**
Documento descreve módulos e fluxo de dados.

**Requisitos funcionais:**

* Explicar componentes.
* Incluir diagrama textual.
* Mapear arquivos a responsabilidades.

**Requisitos não funcionais:**

* Linguagem técnica e objetiva.

**Critérios de aceitação:**

* Dado o repositório.
* Quando abrir `architecture.md`.
* Então o fluxo ponta a ponta deve estar claro.

**Evidências esperadas:**

* `docs/architecture.md`.

**Arquivos impactados:**

* `docs/architecture.md`

**Métricas geradas:**

* Nenhuma.

**Dependências:**

* Estrutura do repositório.

**Riscos:**

* Documento não refletir código.

**Definition of Done:**

* Cada script principal é citado no documento.

---

### US-048 — Criar plano experimental

**Épico:** EPIC-11
**Como:** pesquisador
**Quero:** plano experimental explícito
**Para:** justificar baseline, fine-tuning e MMLU

**Contexto técnico:**
Documento conecta fases do PDF aos experimentos.

**Requisitos funcionais:**

* Descrever baseline.
* Descrever exp A e B.
* Descrever avaliação MMLU.
* Descrever contaminação.

**Requisitos não funcionais:**

* Separar metodologia de resultados.

**Critérios de aceitação:**

* Dado `experiment_plan.md`.
* Quando revisar o plano.
* Então as cinco fases do PDF devem estar cobertas.

**Evidências esperadas:**

* `docs/experiment_plan.md`.

**Arquivos impactados:**

* `docs/experiment_plan.md`

**Métricas geradas:**

* Nenhuma.

**Dependências:**

* Matriz de rastreabilidade.

**Riscos:**

* Omissão de contaminação de dados.

**Definition of Done:**

* Plano cita Spider, MMLU, LoRA/QLoRA e seeds.

---

### US-049 — Criar checklist de reprodutibilidade

**Épico:** EPIC-11
**Como:** avaliador
**Quero:** checklist verificável
**Para:** confirmar que os resultados podem ser reproduzidos

**Contexto técnico:**
Checklist aponta arquivos de evidência.

**Requisitos funcionais:**

* Listar seed.
* Listar hashes.
* Listar versões.
* Listar comandos.

**Requisitos não funcionais:**

* Itens marcáveis.

**Critérios de aceitação:**

* Dado pipeline concluído.
* Quando preencher checklist.
* Então cada item deve apontar para um arquivo.

**Evidências esperadas:**

* `docs/reproducibility_checklist.md`.

**Arquivos impactados:**

* `docs/reproducibility_checklist.md`

**Métricas geradas:**

* `requirements_lock_hash`
* `dataset_hash`

**Dependências:**

* US-002, US-004, US-005.

**Riscos:**

* Hash ausente em alguma etapa.

**Definition of Done:**

* Checklist não contém item sem evidência.

---

### US-050 — Gerar artefatos para relatório

**Épico:** EPIC-11
**Como:** equipe do projeto
**Quero:** assets prontos para relatório
**Para:** reduzir retrabalho manual

**Contexto técnico:**
Relatório PDF deve ter metodologia, resultados, discussão e análise de erros.

**Requisitos funcionais:**

* Exportar tabelas Markdown.
* Exportar resumo de metodologia.
* Exportar análise de erros.
* Exportar discussão de contaminação.

**Requisitos não funcionais:**

* Conteúdo compacto para 10 páginas.

**Critérios de aceitação:**

* Dado pipeline completo.
* Quando `export_report_tables.py` rodar.
* Então todos os assets devem estar em `outputs/report_assets/`.

**Evidências esperadas:**

* `report_outline.md`.
* `error_analysis.md`.
* `contamination_discussion.md`.

**Arquivos impactados:**

* `scripts/export_report_tables.py`
* `docs/`

**Métricas geradas:**

* Nenhuma nova.

**Dependências:**

* US-044, US-045.

**Riscos:**

* Resultados incompletos.

**Definition of Done:**

* Pasta `report_assets/` contém todos os arquivos esperados.

---

## 8. Métricas automatizadas

### Text-to-SQL

* `execution_accuracy`
* `total_examples`
* `correct_examples`
* `failed_examples`
* `invalid_sql_rate`
* `sql_extraction_failure_rate`
* `sqlite_execution_error_rate`
* `average_generation_tokens`
* `average_prompt_tokens`
* `average_latency_seconds`
* `timeout_rate`

### MMLU

* `mmlu_accuracy_overall`
* `mmlu_accuracy_stem`
* `mmlu_accuracy_humanities`
* `mmlu_accuracy_social_sciences`
* `mmlu_delta_overall_percent`
* `mmlu_delta_stem_percent`
* `mmlu_delta_humanities_percent`
* `mmlu_delta_social_sciences_percent`
* `mmlu_parse_error_rate`

### Treinamento

* `train_loss`
* `eval_loss`, se configurado.
* `learning_rate`
* `num_train_epochs`
* `batch_size`
* `gradient_accumulation_steps`
* `lora_r`
* `lora_alpha`
* `lora_dropout`
* `target_modules`
* `quantization_mode`
* `training_runtime_seconds`
* `peak_gpu_memory_mb`
* `tokens_per_second`

### Infraestrutura

* `gpu_name`
* `gpu_vram`
* `cuda_version`
* `torch_version`
* `python_version`
* `runtime_start_time`
* `runtime_end_time`

### Reprodutibilidade

* `seed`
* `model_name`
* `model_revision`
* `dataset_hash`
* `prompt_template_hash`
* `config_hash`
* `requirements_lock_hash`
* `fewshot_examples_hash`
* `mmlu_suite_hash`

### Qualidade de geração

* `average_prompt_tokens`
* `average_generation_tokens`
* `empty_output_rate`
* `non_sql_output_rate`
* `mmlu_parse_error_rate`

### Análise de erros

* `NO_SQL_EXTRACTED`
* `SQL_SYNTAX_ERROR`
* `SQL_EXECUTION_ERROR`
* `SQL_TIMEOUT`
* `RESULT_MISMATCH`
* `DB_NOT_FOUND`
* `UNKNOWN_ERROR`

---

## 9. Configurações editáveis

### `configs/model.yaml`

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-3B"
  trust_remote_code: true
  torch_dtype: "auto"
  load_in_4bit: true
  device_map: "auto"
  use_chat_template_if_available: true
  pad_token_policy: "use_eos_if_missing"
```

### `configs/eval_spider.yaml`

```yaml
evaluation:
  split: "dev"
  batch_size: 1
  limit: null
  timeout_seconds: 30
  output_dir: "outputs/baseline"

fewshot:
  source_split: "train"
  example_ids:
    - "train_0001"
    - "train_0100"
    - "train_0250"

generation:
  max_new_tokens: 256
  do_sample: false
  temperature: 0.0
  top_p: 1.0
  num_beams: 1
  repetition_penalty: 1.0
```

### `configs/train_exp_a.yaml`

```yaml
train:
  experiment_name: "exp_a_lr_2e_4_epochs_1"
  learning_rate: 0.0002
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_strategy: "epoch"
  output_dir: "outputs/finetuned_exp_a"

qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### `configs/train_exp_b.yaml`

```yaml
train:
  experiment_name: "exp_b_lr_1e_4_epochs_2"
  learning_rate: 0.0001
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_strategy: "epoch"
  output_dir: "outputs/finetuned_exp_b"
```

### `configs/eval_mmlu.yaml`

```yaml
mmlu:
  dataset_name: "cais/mmlu"
  total_questions: 150
  seed: 42
  shots: 5
  categories:
    stem:
      subcategory: "college_computer_science"
      eval_count: 50
    humanities:
      subcategory: "philosophy"
      eval_count: 50
    social_sciences:
      subcategory: "econometrics"
      eval_count: 50

generation:
  max_new_tokens: 8
  do_sample: false
  temperature: 0.0
  top_p: 1.0
  num_beams: 1
```

### `requirements.txt`

Versões iniciais sugeridas. Após o primeiro smoke test, gerar `requirements.lock.txt` com o ambiente real.

```text
deepeval==4.0.5
transformers
accelerate
datasets==5.0.0
trl==1.5.1
peft==0.19.1
bitsandbytes
torch
numpy
pandas
pyyaml
tqdm
pytest
sqlparse
safetensors
sentencepiece
```

O DeepEval, TRL, Datasets e PEFT têm versões recentes publicadas no PyPI. ([PyPI][6])

---

## 10. Estratégia de testes

### Testes unitários

* `test_sql_extraction.py`

  * Markdown SQL.
  * Texto antes/depois da query.
  * Prefixo `SQL:`.
  * Saída sem SQL.

* `test_execution_accuracy.py`

  * SQL correta retorna `1.0`.
  * SQL errada retorna `0.0`.
  * Erro de sintaxe retorna `SQL_SYNTAX_ERROR`.
  * `ORDER BY` respeita ordem.
  * Sem `ORDER BY` ignora ordem.

* `test_schema_serialization.py`

  * Schema contém tabelas.
  * Schema contém colunas.
  * Schema contém PK/FK.
  * Ordem é determinística.

* `test_reproducibility.py`

  * Mesma seed gera mesma seleção MMLU.
  * Mesmo template gera mesmo hash.
  * Config hash muda quando YAML muda.

### Testes de integração

* Rodar baseline com `--limit 5`.
* Treinar QLoRA com `--max_steps 2`.
* Avaliar adapter de smoke em 5 exemplos.
* Gerar tabelas com dados mínimos.

### Testes de regressão

* Comparar hash de prompt.
* Comparar hash de dataset.
* Verificar que Spider dev não entra no treino.
* Verificar que MMLU usa exatamente 150 questões.

### Testes da métrica customizada

* Banco SQLite sintético.
* Consultas equivalentes.
* Consultas com duplicatas.
* Consultas com `ORDER BY`.
* Query com timeout.

### Testes de reprodutibilidade

* Duas execuções com seed fixa.
* Mesmos IDs few-shot.
* Mesmos IDs MMLU.
* Mesmo config hash.

### Testes de smoke no Colab

```bash
pytest -q
python scripts/prepare_spider.py --config configs/data.yaml --limit 20
python scripts/run_baseline_spider.py --config configs/eval_spider.yaml --limit 5
python scripts/train_qlora.py --config configs/train_exp_a.yaml --max_steps 2
python scripts/evaluate_mmlu.py --config configs/eval_mmlu.yaml --limit 6
```

---

## 11. Plano de execução no Colab

```bash
git clone <repo-url>
cd project-root
```

```bash
pip install -r requirements.txt
pip freeze > requirements.lock.txt
```

```bash
python -m pytest -q
```

Login Hugging Face, se necessário:

```bash
huggingface-cli login
```

Preparação Spider:

```bash
python scripts/prepare_spider.py \
  --config configs/data.yaml \
  --split train

python scripts/prepare_spider.py \
  --config configs/data.yaml \
  --split dev
```

Preparação MMLU:

```bash
python scripts/prepare_mmlu.py \
  --config configs/eval_mmlu.yaml
```

Baseline Spider:

```bash
python scripts/run_baseline_spider.py \
  --config configs/eval_spider.yaml \
  --output_dir outputs/baseline
```

Fine-tuning experimento A:

```bash
python scripts/train_qlora.py \
  --config configs/train_exp_a.yaml
```

Fine-tuning experimento B:

```bash
python scripts/train_qlora.py \
  --config configs/train_exp_b.yaml
```

Avaliação Spider dos fine-tuned:

```bash
python scripts/evaluate_spider.py \
  --config configs/eval_spider.yaml \
  --adapter_path outputs/finetuned_exp_a/adapters \
  --output_dir outputs/finetuned_exp_a

python scripts/evaluate_spider.py \
  --config configs/eval_spider.yaml \
  --adapter_path outputs/finetuned_exp_b/adapters \
  --output_dir outputs/finetuned_exp_b
```

Avaliação MMLU:

```bash
python scripts/evaluate_mmlu.py \
  --config configs/eval_mmlu.yaml \
  --model_id base \
  --output_path outputs/mmlu/mmlu_baseline.json

python scripts/evaluate_mmlu.py \
  --config configs/eval_mmlu.yaml \
  --model_id exp_a \
  --adapter_path outputs/finetuned_exp_a/adapters \
  --output_path outputs/mmlu/mmlu_exp_a.json

python scripts/evaluate_mmlu.py \
  --config configs/eval_mmlu.yaml \
  --model_id exp_b \
  --adapter_path outputs/finetuned_exp_b/adapters \
  --output_path outputs/mmlu/mmlu_exp_b.json
```

Agregação e exportação:

```bash
python scripts/aggregate_results.py
python scripts/export_report_tables.py
```

---

## 12. Artefatos finais esperados

```text
outputs/baseline/spider_predictions.jsonl
outputs/baseline/spider_metrics.json
outputs/baseline/model_metadata.json
outputs/finetuned_exp_a/adapters/
outputs/finetuned_exp_a/spider_predictions.jsonl
outputs/finetuned_exp_a/spider_metrics.json
outputs/finetuned_exp_a/train_metrics.json
outputs/finetuned_exp_b/adapters/
outputs/finetuned_exp_b/spider_predictions.jsonl
outputs/finetuned_exp_b/spider_metrics.json
outputs/finetuned_exp_b/train_metrics.json
outputs/mmlu/mmlu_suite.jsonl
outputs/mmlu/mmlu_baseline.json
outputs/mmlu/mmlu_exp_a.json
outputs/mmlu/mmlu_exp_b.json
outputs/metrics/environment.json
outputs/metrics/reproducibility.json
outputs/metrics/comparison_table.csv
outputs/metrics/training_summary.csv
outputs/report_assets/spider_results_table.md
outputs/report_assets/mmlu_regression_table.md
outputs/report_assets/error_analysis.md
outputs/report_assets/contamination_discussion.md
outputs/report_assets/report_outline.md
docs/user_stories.md
docs/architecture.md
docs/traceability_matrix.md
docs/experiment_plan.md
docs/reproducibility_checklist.md
README.md
requirements.txt
requirements.lock.txt
```

---

## 13. Checklist de conformidade

| Item                                               | Evidência esperada                             | Status  |
| -------------------------------------------------- | ---------------------------------------------- | ------- |
| Spider train usado exclusivamente para fine-tuning | `split_audit.json`                             | Coberto |
| Spider dev usado apenas para avaliação             | `spider_dev_eval.jsonl` e validação do trainer | Coberto |
| Prompt Text-to-SQL contém schema do banco          | `prompt_render_samples.jsonl`                  | Coberto |
| Baseline usa prompt few-shot fixo com 3 exemplos   | `fewshot_examples_hash`                        | Coberto |
| Execution Accuracy é única métrica Text-to-SQL     | `spider_metrics.json`                          | Coberto |
| Métrica é usada no baseline e fine-tuned           | mesmo `evaluate_spider.py`                     | Coberto |
| DeepEval é usado na métrica customizada            | `execution_accuracy.py`                        | Coberto |
| LoRA ou QLoRA é usado                              | `adapter_config.json`                          | Coberto |
| Duas configurações de hiperparâmetros são testadas | `train_exp_a.yaml`, `train_exp_b.yaml`         | Coberto |
| Hardware é documentado                             | `environment.json`                             | Coberto |
| MMLU usa exatamente 150 questões                   | `mmlu_suite.jsonl`                             | Coberto |
| MMLU usa 5-shot                                    | `mmlu_fewshot_examples.json`                   | Coberto |
| MMLU compara base vs fine-tuned                    | `mmlu_regression_table.md`                     | Coberto |
| Regressão é calculada por categoria e agregada     | `mmlu_delta_*`                                 | Coberto |
| Seeds são fixas                                    | `reproducibility.json`                         | Coberto |
| Geração é determinística                           | `generation_config.json`                       | Coberto |
| Dependências têm versões fixadas                   | `requirements.txt`, `requirements.lock.txt`    | Coberto |
| README explica reprodução                          | `README.md`                                    | Coberto |
| Relatório tem metodologia, resultados e discussão  | `report_outline.md`                            | Coberto |
| Existe discussão sobre contaminação de dados       | `contamination_discussion.md`                  | Coberto |
| Métricas são exportadas automaticamente            | `outputs/metrics/*.csv/json`                   | Coberto |
| Arquitetura reduz retrabalho manual                | `report_assets/` e scripts de exportação       | Coberto |

---

## 14. Riscos e mitigações

| Risco                                        | Impacto                               | Mitigação prática                                                                                     |
| -------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Falta de VRAM na T4                          | OOM no carregamento ou treino         | Usar QLoRA 4-bit, batch 1, gradient checkpointing e `max_seq_length=2048`.                            |
| Tempo excessivo no Colab                     | Sessão cair antes do fim              | Salvar checkpoints por época e escrever JSONL incremental.                                            |
| Modelo não-instruct                          | Saídas menos obedientes ao prompt     | Registrar no relatório; se exigido, trocar para `Qwen/Qwen2.5-Coder-3B-Instruct`.                     |
| Incompatibilidade de módulos LoRA            | Treino falha ao aplicar adapters      | Detectar módulos existentes e salvar `lora_modules_report.json`.                                      |
| Erro de extração SQL                         | Execution Accuracy subestimada        | Testar extração com markdown, prefixos, texto extra e ausência de SQL.                                |
| SQL lenta ou maliciosa                       | Avaliação trava                       | SQLite read-only, timeout e `PRAGMA query_only=ON`.                                                   |
| Diferença frente ao avaliador oficial Spider | Resultados podem divergir             | Documentar limitação e aproximar semântica de ordem e multiset.                                       |
| Contaminação de dados                        | Interpretação dos ganhos fica ambígua | Criar seção específica no relatório; tratar resultados como empíricos, não prova de aprendizado puro. |
| Instabilidade de dependências                | Reprodução falha                      | Pin em `requirements.txt` e hash de `requirements.lock.txt`.                                          |
| Falha de download dos datasets               | Pipeline não inicia                   | Permitir montagem manual em `data/raw/` e validar arquivos esperados.                                 |
| MMLU com subcategoria inexistente            | Preparação falha                      | Validar configs disponíveis e interromper com mensagem clara.                                         |
| Colab sem GPU                                | Treino inviável                       | Permitir smoke tests CPU; bloquear treino completo sem GPU.                                           |

---

## 15. Recomendações finais

1. **Começar pela métrica Execution Accuracy.** Ela é o eixo do projeto. Sem ela, baseline e fine-tuning não são comparáveis.

2. **Implementar primeiro o modo smoke.** Use `--limit 5` para Spider, `--max_steps 2` para treino e `--limit 6` para MMLU.

3. **Nunca misturar Spider train e dev.** O script de treino deve falhar se receber qualquer arquivo marcado como dev.

4. **Salvar tudo incrementalmente.** Em Colab, quedas de runtime são comuns. JSONL por exemplo evita perder uma avaliação inteira.

5. **Validar o modelo-alvo com o professor.** O PDF pede instruct/chat. O modelo solicitado é `Qwen/Qwen2.5-Coder-3B`. A troca para `Qwen/Qwen2.5-Coder-3B-Instruct` é a menor alteração caso a exigência seja rígida.

6. **Separar métrica principal de métricas diagnósticas.** Execution Accuracy decide desempenho Text-to-SQL. Taxas de erro, latência e tokens explicam causas.

7. **Gerar as tabelas automaticamente.** A equipe deve escrever o relatório a partir de `outputs/report_assets/`, não a partir de inspeção manual dos logs.

[1]: https://huggingface.co/Qwen/Qwen2.5-Coder-3B?utm_source=chatgpt.com "Qwen/Qwen2.5-Coder-3B"
[2]: https://yale-lily.github.io/spider?utm_source=chatgpt.com "Spider: Yale Semantic Parsing and Text-to-SQL Challenge"
[3]: https://huggingface.co/datasets/cais/mmlu?utm_source=chatgpt.com "cais/mmlu · Datasets at Hugging Face"
[4]: https://deepeval.com/guides/guides-building-custom-metrics?utm_source=chatgpt.com "Building Custom LLM Metrics | DeepEval by Confident AI"
[5]: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes?utm_source=chatgpt.com "Bitsandbytes"
[6]: https://pypi.org/project/deepeval/?utm_source=chatgpt.com "deepeval"

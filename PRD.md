# Documentação técnica de implementação — TP2 NLP/Text-to-SQL

**Fonte principal:** enunciado do TP2 anexado e decisões práticas fornecidas na conversa. O enunciado define que o trabalho deve medir empiricamente o trade-off entre especialização em Text-to-SQL e possível regressão em tarefas gerais, usando Spider, MMLU, LoRA/QLoRA, DeepEval e reprodutibilidade estrita.

---

## 0. Decisões consolidadas, conflitos e assunções

### 0.1 Decisões obrigatórias vindas do enunciado

| Tema                  | Decisão obrigatória                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------- |
| Tarefa-alvo           | Text-to-SQL usando Spider                                                                     |
| Treinamento           | Usar apenas o **training split** do Spider                                                    |
| Avaliação Text-to-SQL | Usar o **development split** do Spider                                                        |
| Métrica Text-to-SQL   | `Execution Accuracy` customizada em DeepEval                                                  |
| Generalização         | Avaliar no MMLU com exatamente 150 questões                                                   |
| MMLU                  | 50 STEM, 50 Humanidades, 50 Ciências Sociais                                                  |
| Fine-tuning           | Usar LoRA; QLoRA é permitido e recomendado para pouca VRAM                                    |
| Experimentos          | Testar pelo menos 2 configurações de hiperparâmetros                                          |
| Reprodutibilidade     | Fixar seeds e usar geração determinística                                                     |
| Entregáveis           | Repositório GitHub, `requirements.txt`, `README.md`, relatório PDF IEEE/ACM de até 10 páginas |

O enunciado também exige que o prompt contenha o esquema do banco, porque o modelo precisa conhecer tabelas, colunas e chaves para gerar SQL válida.

### 0.2 Decisões do projeto

| Tema                     | Decisão adotada                                        |
| ------------------------ | ------------------------------------------------------ |
| Modelo principal         | `Qwen/Qwen2.5-3B-Instruct`                             |
| Estratégia principal     | LoRA em BF16/FP16 no perfil L4                         |
| Estratégia fallback      | QLoRA 4-bit no perfil T4                               |
| Experimento A            | Conservador: LR `1e-4`, 1 época                        |
| Experimento B            | Mais agressivo: LR `2e-4`, 2 épocas                    |
| Comparação principal     | Base vs LoRA Exp A vs LoRA Exp B                       |
| Prompt Spider            | Template único para baseline e fine-tuned              |
| Avaliação MMLU           | 5-shot fixo, greedy, temperatura 0                     |
| Ordem de desenvolvimento | Avaliação primeiro, treinamento depois                 |
| Notebook Colab           | Apenas orquestra comandos; não contém lógica principal |

O modelo escolhido está dentro da classe de 3–4B exigida pelo enunciado. O model card oficial do Qwen2.5 informa que a família Qwen2.5 possui modelos base e instruct em múltiplos tamanhos, incluindo a variante 3B. ([Hugging Face][1])

### 0.3 Conflito identificado: T4 vs L4

**Conflito:** o enunciado pede adequação ao Google Colab gratuito com GPU T4, enquanto as decisões práticas pedem “altíssima performance” na GPU L4 do Google Colab.

**Impacto técnico:** a L4 normalmente permite batch maior, sequência maior e execução LoRA sem quantização agressiva. A T4 exige economia de VRAM. Usar uma única configuração para as duas GPUs pode gerar OOM na T4 ou subutilizar a L4.

**Interpretação recomendada:**
O projeto deve ter **dois perfis de execução**:

1. `profile: l4` — perfil principal, otimizado para performance.
2. `profile: t4` — fallback, otimizado para memória.

O protocolo experimental não muda. Só mudam parâmetros operacionais como quantização, batch size, gradient accumulation e comprimento máximo de sequência.

### 0.4 Assunções técnicas

| Item                               | Assunção                                                                                                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset MMLU                       | Usar `cais/mmlu` no Hugging Face Hub, salvo indisponibilidade no ambiente. Esse dataset cobre múltiplas áreas e tarefas do MMLU. ([Hugging Face][2]) |
| Subcategorias MMLU                 | Fixar `college_computer_science`, `philosophy` e `econometrics`, salvo orientação diferente do professor.                                            |
| Tipo de SQL                        | Considerar consultas Spider como consultas analíticas `SELECT`/`WITH`; bloquear comandos destrutivos na execução da SQL prevista.                    |
| Relatório                          | O relatório final apresentará metodologia, resultados, análise de erro e discussão de contaminação.                                                  |
| Controle de versão de dependências | As versões exatas devem ser fixadas no `requirements.txt` após validação no ambiente Colab usado para gerar os resultados finais.                    |

---

# 1. Visão geral do sistema

O sistema é um **pipeline experimental reprodutível de fine-tuning e avaliação de LLMs**. Ele mede dois efeitos:

1. **Especialização:** ganho de desempenho em Text-to-SQL após fine-tuning no Spider.
2. **Generalização:** possível perda de desempenho em tarefas gerais após fine-tuning, medida pelo MMLU.

A analogia correta é: o projeto funciona como uma **bancada de laboratório**. O modelo base é o material original. Os experimentos A e B são tratamentos diferentes. O Spider mede se o tratamento melhorou a tarefa específica. O MMLU mede se o tratamento causou efeitos colaterais.

## Objetivo técnico

Implementar um repositório capaz de:

* Preparar Spider e MMLU.
* Montar prompts determinísticos.
* Rodar baseline no modelo base.
* Treinar dois adapters LoRA/QLoRA.
* Avaliar Spider por execução SQL.
* Avaliar MMLU por acurácia.
* Gerar métricas, predições, logs e resumo comparativo.
* Permitir reprodução local e no Google Colab.

## Público-alvo

* Desenvolvedor responsável pela implementação.
* Pesquisador/aluno responsável pelo experimento.
* Avaliador/professor que precisa reproduzir os resultados.
* QA técnico que valida métricas, scripts e entregáveis.

---

# 2. Escopo do projeto

## 2.1 Dentro do escopo

### Funcionalidades principais

| ID    | Funcionalidade                                        |
| ----- | ----------------------------------------------------- |
| ES-01 | Preparar dados do Spider para treinamento e avaliação |
| ES-02 | Preparar suíte fixa de 150 questões do MMLU           |
| ES-03 | Implementar prompt few-shot fixo para Spider          |
| ES-04 | Implementar prompt 5-shot fixo para MMLU              |
| ES-05 | Implementar métrica `ExecutionAccuracy` em DeepEval   |
| ES-06 | Avaliar modelo base no Spider dev                     |
| ES-07 | Avaliar modelo base no MMLU 150                       |
| ES-08 | Treinar LoRA/QLoRA Experimento A                      |
| ES-09 | Treinar LoRA/QLoRA Experimento B                      |
| ES-10 | Avaliar modelos fine-tuned no Spider dev              |
| ES-11 | Avaliar modelos fine-tuned no MMLU                    |
| ES-12 | Gerar arquivos JSONL de predição                      |
| ES-13 | Gerar métricas agregadas e por categoria              |
| ES-14 | Gerar `summary.md` por experimento                    |
| ES-15 | Registrar ambiente, seeds, GPU, versões e configs     |

### Funcionalidades secundárias

| ID    | Funcionalidade                                   |
| ----- | ------------------------------------------------ |
| ES-16 | Atalhos via `Makefile`                           |
| ES-17 | Notebook Colab para execução dos comandos        |
| ES-18 | Testes unitários da métrica                      |
| ES-19 | Testes de integração com banco SQLite artificial |
| ES-20 | Smoke test com subconjunto pequeno               |

### Evoluções futuras

| ID     | Evolução                                               |
| ------ | ------------------------------------------------------ |
| FUT-01 | Adicionar avaliação oficial Spider test-suite completa |
| FUT-02 | Comparar Qwen 3B contra outro modelo 3B                |
| FUT-03 | Adicionar análise por dificuldade da SQL               |
| FUT-04 | Adicionar dashboard simples com resultados             |
| FUT-05 | Adicionar suporte opcional a W&B ou MLflow             |

## 2.2 Fora do escopo

* Servir o modelo em produção.
* Criar API HTTP pública.
* Criar interface web.
* Usar LLM-as-a-judge para Text-to-SQL.
* Usar BLEU, ROUGE ou similaridade textual de SQL.
* Fazer busca ampla de hiperparâmetros.
* Usar Spider dev durante treinamento.
* Alterar prompts entre baseline e modelos fine-tuned.
* Avaliar com temperatura maior que zero.

---

# 3. Requisitos funcionais

## 3.1 Preparação de dados

| ID    | Requisito                                                                                         |
| ----- | ------------------------------------------------------------------------------------------------- |
| RF-01 | O sistema deve preparar o Spider a partir de diretórios configuráveis via argumento ou YAML.      |
| RF-02 | O sistema deve separar claramente Spider train e Spider dev.                                      |
| RF-03 | O Spider train deve ser usado somente para fine-tuning e exemplos few-shot.                       |
| RF-04 | O Spider dev deve ser usado somente para avaliação final de Text-to-SQL.                          |
| RF-05 | O sistema deve carregar os bancos SQLite do Spider por `db_id`.                                   |
| RF-06 | O sistema deve extrair e serializar o esquema de cada banco com tabelas, colunas, tipos e chaves. |
| RF-07 | O sistema deve preparar uma suíte MMLU fixa com exatamente 150 questões.                          |
| RF-08 | A suíte MMLU deve conter 50 questões de STEM, 50 de Humanidades e 50 de Ciências Sociais.         |
| RF-09 | A seleção da suíte MMLU deve ser determinística por seed.                                         |
| RF-10 | O arquivo final da suíte MMLU deve ser salvo em `data/processed/mmlu_150_suite.json`.             |

## 3.2 Prompting

| ID    | Requisito                                                                                                |
| ----- | -------------------------------------------------------------------------------------------------------- |
| RF-11 | O sistema deve implementar `build_spider_prompt(example, few_shot_examples)`.                            |
| RF-12 | O prompt Spider deve conter instrução da tarefa, esquema do banco, 3 exemplos few-shot e pergunta final. |
| RF-13 | O prompt Spider deve instruir o modelo a responder apenas com SQL.                                       |
| RF-14 | O mesmo prompt Spider deve ser usado para baseline, Exp A e Exp B.                                       |
| RF-15 | O sistema deve implementar prompt MMLU 5-shot.                                                           |
| RF-16 | Os 5 exemplos de contexto do MMLU devem ser fixos por subcategoria.                                      |
| RF-17 | O prompt MMLU deve solicitar resposta única entre `A`, `B`, `C` ou `D`.                                  |

## 3.3 Inferência

| ID    | Requisito                                                                        |
| ----- | -------------------------------------------------------------------------------- |
| RF-18 | O sistema deve carregar o modelo base sem adapter para baseline.                 |
| RF-19 | O sistema deve carregar modelo base + adapter LoRA para experimentos fine-tuned. |
| RF-20 | A geração de avaliação deve usar `temperature=0`.                                |
| RF-21 | A geração de avaliação deve usar decoding greedy.                                |
| RF-22 | O sistema deve registrar a saída bruta do modelo antes da extração da SQL.       |
| RF-23 | O sistema deve registrar a SQL extraída após limpeza.                            |
| RF-24 | O sistema deve registrar falhas de parsing no MMLU como erro.                    |

## 3.4 Métrica Text-to-SQL

| ID    | Requisito                                                                                  |
| ----- | ------------------------------------------------------------------------------------------ |
| RF-25 | O sistema deve implementar `ExecutionAccuracy` em `custom_metrics/execution_accuracy.py`.  |
| RF-26 | A métrica deve herdar de `deepeval.metrics.BaseMetric`.                                    |
| RF-27 | A métrica deve implementar `measure(self, test_case: LLMTestCase) -> float`.               |
| RF-28 | A métrica deve extrair SQL da saída bruta removendo markdown e explicações.                |
| RF-29 | A métrica deve executar SQL prevista e SQL gold no SQLite correto.                         |
| RF-30 | A métrica deve retornar `1.0` quando os resultados forem equivalentes.                     |
| RF-31 | A métrica deve retornar `0.0` em erro de sintaxe, erro de execução ou resultado diferente. |
| RF-32 | A comparação deve ignorar ordem quando a query não tiver `ORDER BY`.                       |
| RF-33 | A comparação deve preservar ordem quando a query tiver `ORDER BY`.                         |
| RF-34 | A métrica deve ser aplicada igualmente no baseline e nos modelos fine-tuned.               |

O enunciado exige que a métrica herde de `BaseMetric`, implemente `measure` e compare resultados executados no SQLite, retornando `1.0` ou `0.0`.  A documentação oficial do DeepEval também orienta que métricas customizadas herdem de `BaseMetric`. ([DeepEval][3])

## 3.5 Treinamento

| ID    | Requisito                                                                               |
| ----- | --------------------------------------------------------------------------------------- |
| RF-35 | O treinamento deve ser executado por `python -m scripts.train --config <arquivo.yaml>`. |
| RF-36 | O script deve ler todos os hiperparâmetros do YAML.                                     |
| RF-37 | O script deve fixar seeds antes de carregar dados e modelo.                             |
| RF-38 | O script deve carregar modelo e tokenizer.                                              |
| RF-39 | O script deve formatar exemplos Spider train para SFT.                                  |
| RF-40 | O script deve aplicar LoRA ou QLoRA conforme perfil de GPU.                             |
| RF-41 | O script deve salvar adapter, tokenizer e config final.                                 |
| RF-42 | O script deve salvar logs de treinamento.                                               |
| RF-43 | O script deve suportar Experimento A e Experimento B.                                   |

A biblioteca TRL oferece `SFTTrainer` para treinamento supervisionado e aceita datasets nos formatos de linguagem, prompt-completion e conversacional. ([Hugging Face][4]) A integração TRL + PEFT suporta LoRA/QLoRA via `peft_config`. ([Hugging Face][5])

## 3.6 Avaliação

| ID    | Requisito                                                                                                                           |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------- |
| RF-44 | O comando principal de avaliação deve ser `python -m scripts.run_benchmarks --config configs/eval.yaml --model_path outputs/<exp>`. |
| RF-45 | O benchmark deve executar Spider dev e MMLU 150.                                                                                    |
| RF-46 | O benchmark deve calcular métricas do modelo avaliado.                                                                              |
| RF-47 | O benchmark deve calcular variação contra o modelo base quando os resultados base existirem.                                        |
| RF-48 | O benchmark deve salvar `spider_predictions.jsonl`.                                                                                 |
| RF-49 | O benchmark deve salvar `spider_metrics.json`.                                                                                      |
| RF-50 | O benchmark deve salvar `mmlu_predictions.jsonl`.                                                                                   |
| RF-51 | O benchmark deve salvar `mmlu_metrics.json`.                                                                                        |
| RF-52 | O benchmark deve salvar `summary.md`.                                                                                               |

---

# 4. Requisitos não funcionais

## 4.1 Reprodutibilidade

| ID     | Requisito                                                                             |
| ------ | ------------------------------------------------------------------------------------- |
| RNF-01 | Todos os scripts devem aceitar paths por argumento ou YAML.                           |
| RNF-02 | O código não deve conter paths absolutos de Colab ou Windows.                         |
| RNF-03 | Seeds devem ser fixadas para Python, NumPy, PyTorch, Hugging Face e amostragem.       |
| RNF-04 | A suíte MMLU deve ser persistida para evitar reamostragem acidental.                  |
| RNF-05 | Geração de avaliação deve usar `temperature=0` e greedy decoding.                     |
| RNF-06 | Todas as configs finais devem ser copiadas para `outputs/<exp>/training_config.yaml`. |
| RNF-07 | O ambiente deve registrar GPU, VRAM, data da execução e versões das bibliotecas.      |

O enunciado exige reprodutibilidade, fixação de seeds e geração determinística para avaliação.

## 4.2 Performance

| ID     | Requisito                                                            |
| ------ | -------------------------------------------------------------------- |
| RNF-08 | O perfil L4 deve priorizar velocidade de treinamento e avaliação.    |
| RNF-09 | O perfil T4 deve priorizar baixo consumo de VRAM.                    |
| RNF-10 | O pipeline deve permitir limitar número de exemplos para smoke test. |
| RNF-11 | A avaliação deve usar batch controlado para evitar OOM.              |
| RNF-12 | O carregamento de modelo deve usar cache configurável.               |

## 4.3 Segurança operacional

| ID     | Requisito                                                                                        |
| ------ | ------------------------------------------------------------------------------------------------ |
| RNF-13 | SQL gerada pelo modelo deve ser executada apenas contra cópia local/read-only dos bancos Spider. |
| RNF-14 | A métrica deve rejeitar comandos SQL destrutivos.                                                |
| RNF-15 | Tokens Hugging Face, se usados, não devem ser versionados.                                       |
| RNF-16 | Outputs grandes não devem ser commitados.                                                        |

## 4.4 Manutenibilidade

| ID     | Requisito                                                              |
| ------ | ---------------------------------------------------------------------- |
| RNF-17 | A lógica principal deve ficar em `src/tp2/`.                           |
| RNF-18 | Scripts em `scripts/` devem orquestrar módulos, não concentrar regras. |
| RNF-19 | Funções críticas devem ter testes unitários.                           |
| RNF-20 | Configurações devem ser declarativas em YAML.                          |
| RNF-21 | Logs e resultados devem ter schema estável.                            |

## 4.5 Observabilidade

| ID     | Requisito                                                                     |
| ------ | ----------------------------------------------------------------------------- |
| RNF-22 | Cada predição deve ser rastreável por `example_id` ou índice.                 |
| RNF-23 | Erros devem registrar tipo e mensagem sanitizada.                             |
| RNF-24 | Métricas devem ser salvas em JSON legível.                                    |
| RNF-25 | O relatório deve conseguir ser produzido a partir dos arquivos em `outputs/`. |

---

# 5. Perfis de usuário e permissões

Como o projeto é um pipeline local/Colab, não há autenticação de aplicação nem papéis em uma API. Mesmo assim, existem perfis operacionais.

| Perfil                  | Responsabilidades                            | Permissões                                       |
| ----------------------- | -------------------------------------------- | ------------------------------------------------ |
| Desenvolvedor           | Implementar módulos, testes e scripts        | Editar código, configs e testes                  |
| Executor do experimento | Rodar preparação, treinamento e avaliação    | Executar scripts e gerar outputs                 |
| Avaliador/Professor     | Reproduzir resultados                        | Ler README, rodar comandos e inspecionar outputs |
| QA técnico              | Validar métricas, logs e critérios de aceite | Rodar testes e smoke tests                       |

## Restrições

* Apenas desenvolvedores devem alterar lógica da métrica.
* O executor não deve editar outputs manualmente.
* O avaliador deve conseguir reproduzir resultados usando README + configs + seeds.
* O notebook Colab não deve conter lógica exclusiva.

---

# 6. Regras de negócio

## 6.1 Regras de dados

| ID    | Regra                                                                         |
| ----- | ----------------------------------------------------------------------------- |
| RN-01 | Spider train pode ser usado para treinamento e few-shot.                      |
| RN-02 | Spider dev não pode ser usado no treinamento.                                 |
| RN-03 | MMLU 150 deve ser fixo para todos os modelos.                                 |
| RN-04 | Os mesmos exemplos few-shot devem ser usados para todos os modelos avaliados. |
| RN-05 | O baseline deve ser sempre o modelo base sem adapter.                         |

## 6.2 Regras de avaliação Spider

| ID    | Regra                                                                  |
| ----- | ---------------------------------------------------------------------- |
| RN-06 | A avaliação Text-to-SQL usa execução, não comparação textual.          |
| RN-07 | SQL prevista e SQL gold devem rodar no mesmo banco SQLite.             |
| RN-08 | Resultado idêntico gera score `1.0`.                                   |
| RN-09 | Resultado diferente gera score `0.0`.                                  |
| RN-10 | Erro de sintaxe gera score `0.0`.                                      |
| RN-11 | Tabela ou coluna inexistente gera score `0.0`.                         |
| RN-12 | Falha de execução gera score `0.0`.                                    |
| RN-13 | Query sem `ORDER BY` deve comparar linhas como conjunto/multiconjunto. |
| RN-14 | Query com `ORDER BY` deve preservar ordem.                             |
| RN-15 | A saída bruta do modelo deve ser salva mesmo quando a extração falhar. |

## 6.3 Regras de avaliação MMLU

| ID    | Regra                                                                             |
| ----- | --------------------------------------------------------------------------------- |
| RN-16 | Cada questão deve ter quatro alternativas.                                        |
| RN-17 | A resposta válida é uma letra entre `A`, `B`, `C`, `D`.                           |
| RN-18 | Falha de parsing conta como erro.                                                 |
| RN-19 | Acurácia geral = acertos / 150.                                                   |
| RN-20 | Acurácia por categoria = acertos da categoria / 50.                               |
| RN-21 | A mesma suíte e os mesmos exemplos 5-shot devem ser usados para todos os modelos. |

## 6.4 Regras de comparação

| ID    | Regra                                                                                         |
| ----- | --------------------------------------------------------------------------------------------- |
| RN-22 | Ganho Spider = `Acc_spider_finetuned - Acc_spider_base`.                                      |
| RN-23 | Variação MMLU = `Acc_mmlu_finetuned - Acc_mmlu_base`.                                         |
| RN-24 | Variação percentual = `((Acc_finetuned - Acc_base) / Acc_base) * 100`, quando `Acc_base > 0`. |
| RN-25 | A conclusão deve ser limitada ao modelo, dados, seeds, prompts e hiperparâmetros usados.      |

---

# 7. Fluxos principais do sistema

## 7.1 Fluxo 1 — Preparar Spider

**Ator:** executor do experimento
**Pré-condições:** Spider baixado ou disponível no diretório configurado.

### Etapas

1. Executar:

```bash
python -m scripts.prepare_spider --data_dir /content/data
```

2. Validar existência de:

   * `train_spider.json`
   * `dev.json`
   * diretório `database/`
3. Ler schemas dos bancos SQLite.
4. Gerar representação textual dos schemas.
5. Salvar artefatos processados em `data/processed/spider/`.

### Resultado esperado

* Dados Spider preparados para treinamento e avaliação.
* Nenhum dado do dev aparece nos arquivos de treinamento.

### Exceções

| Erro                   | Tratamento                                       |
| ---------------------- | ------------------------------------------------ |
| Arquivo Spider ausente | Encerrar com mensagem indicando arquivo esperado |
| Banco SQLite ausente   | Registrar `missing_database`                     |
| Schema inválido        | Registrar `schema_parse_error`                   |

---

## 7.2 Fluxo 2 — Preparar MMLU 150

**Ator:** executor
**Pré-condições:** acesso ao Hugging Face Hub ou cache local do dataset.

### Etapas

1. Executar:

```bash
python -m scripts.prepare_mmlu --config configs/eval.yaml
```

2. Carregar subcategorias fixas.
3. Selecionar 50 questões por categoria.
4. Selecionar exemplos 5-shot fixos.
5. Salvar:

```text
data/processed/mmlu_150_suite.json
```

### Resultado esperado

* Arquivo com 150 questões.
* Questões têm categoria, subcategoria, pergunta, alternativas e resposta gold.

### Exceções

| Erro                 | Tratamento                           |
| -------------------- | ------------------------------------ |
| Dataset indisponível | Orientar uso de cache/local download |
| Menos de 50 questões | Encerrar com erro de configuração    |
| Label inválido       | Registrar `invalid_mmlu_label`       |

---

## 7.3 Fluxo 3 — Avaliar baseline Spider

**Ator:** executor
**Pré-condições:** Spider preparado, modelo base carregável.

### Etapas

1. Executar:

```bash
python -m scripts.evaluate_spider \
  --config configs/eval.yaml \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_dir outputs/base
```

2. Para cada exemplo do Spider dev:

   * Montar prompt com schema + few-shot + pergunta.
   * Gerar saída determinística.
   * Extrair SQL.
   * Executar SQL prevista e SQL gold.
   * Comparar resultados.
   * Salvar predição JSONL.

### Resultado esperado

* `outputs/base/spider_predictions.jsonl`
* `outputs/base/spider_metrics.json`

---

## 7.4 Fluxo 4 — Avaliar baseline MMLU

**Ator:** executor
**Pré-condições:** suíte MMLU preparada.

### Etapas

1. Executar:

```bash
python -m scripts.evaluate_mmlu \
  --config configs/eval.yaml \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_dir outputs/base
```

2. Para cada questão:

   * Montar prompt 5-shot.
   * Gerar resposta.
   * Extrair primeira alternativa válida.
   * Comparar com gold.
   * Salvar predição.

### Resultado esperado

* `outputs/base/mmlu_predictions.jsonl`
* `outputs/base/mmlu_metrics.json`

---

## 7.5 Fluxo 5 — Treinar Experimento A

**Ator:** executor
**Pré-condições:** Spider train preparado.

```bash
python -m scripts.train --config configs/train_lora_exp_a.yaml
```

### Resultado esperado

```text
outputs/exp_a/
├── adapter/
├── tokenizer/
├── training_config.yaml
├── train_logs.json
└── environment.json
```

---

## 7.6 Fluxo 6 — Treinar Experimento B

```bash
python -m scripts.train --config configs/train_lora_exp_b.yaml
```

### Resultado esperado

```text
outputs/exp_b/
├── adapter/
├── tokenizer/
├── training_config.yaml
├── train_logs.json
└── environment.json
```

---

## 7.7 Fluxo 7 — Rodar benchmarks finais

```bash
python -m scripts.run_benchmarks \
  --config configs/eval.yaml \
  --model_path outputs/exp_a
```

Repetir para `outputs/exp_b`.

### Resultado esperado por experimento

```text
outputs/exp_a/
├── adapter/
├── training_config.yaml
├── train_logs.json
├── spider_predictions.jsonl
├── spider_metrics.json
├── mmlu_predictions.jsonl
├── mmlu_metrics.json
└── summary.md
```

---

# 8. Arquitetura proposta

## 8.1 Estilo arquitetural recomendado

A arquitetura recomendada é um **modular monolith orientado a scripts**, com separação em camadas internas.

Esse estilo é adequado porque:

* O projeto é experimental, não um sistema distribuído.
* A equipe precisa de reprodutibilidade e simplicidade.
* Scripts CLI são suficientes.
* Módulos separados reduzem ambiguidade e facilitam testes.
* Evita complexidade desnecessária de microsserviços, filas ou APIs HTTP.

## 8.2 Camadas

```text
scripts/             -> Entrada CLI e orquestração
custom_metrics/      -> Métrica DeepEval
src/tp2/data.py      -> Carregamento e preparação de dados
src/tp2/prompts.py   -> Templates de prompt
src/tp2/model.py     -> Carregamento de modelo/tokenizer/adapters
src/tp2/inference.py -> Geração e parsing de respostas
src/tp2/sql_utils.py -> Execução e comparação SQL
src/tp2/logging_utils.py -> Logs, métricas e ambiente
configs/             -> Parâmetros declarativos
outputs/             -> Artefatos gerados
```

## 8.3 Diagrama lógico

```text
                 ┌──────────────────┐
                 │     configs/     │
                 └────────┬─────────┘
                          │
                          v
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ Spider/MMLU │ -> │ data.py      │ -> │ prompts.py     │
└─────────────┘    └──────────────┘    └───────┬────────┘
                                                │
                                                v
                                      ┌──────────────────┐
                                      │ model.py         │
                                      │ inference.py     │
                                      └───────┬──────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        v                                           v
              ┌──────────────────┐                       ┌──────────────────┐
              │ sql_utils.py     │                       │ MMLU parser      │
              │ ExecutionAccuracy│                       │ accuracy         │
              └────────┬─────────┘                       └────────┬─────────┘
                       v                                          v
              ┌────────────────────────────────────────────────────────┐
              │ outputs/<exp>/*.jsonl, *.json, summary.md              │
              └────────────────────────────────────────────────────────┘
```

## 8.4 Trade-offs

| Opção                | Benefício                                         | Custo                                                           |
| -------------------- | ------------------------------------------------- | --------------------------------------------------------------- |
| Modular monolith     | Simples, testável, fácil de rodar no Colab        | Menos flexível que serviços separados                           |
| YAML configs         | Reprodutível e auditável                          | Requer validação de schema                                      |
| Scripts CLI          | Fácil para entrega acadêmica                      | Sem interface gráfica                                           |
| LoRA L4              | Mais rápido e simples que QLoRA quando há memória | Pode falhar em T4                                               |
| QLoRA T4             | Menor VRAM                                        | Pode ser mais lento e mais sensível a versões CUDA/bitsandbytes |
| Métrica por execução | Avalia equivalência real de resultado             | Mais lenta e exige cuidado com SQLite                           |

---

# 9. Organização dos módulos

## 9.1 `src/tp2/data.py`

**Responsabilidade:** carregar, validar e preparar dados.

### Funções esperadas

```python
load_spider_train(data_dir: Path) -> list[SpiderExample]
load_spider_dev(data_dir: Path) -> list[SpiderExample]
load_spider_schema(db_path: Path) -> SpiderSchema
serialize_schema(schema: SpiderSchema) -> str
load_mmlu_suite(path: Path) -> list[MMLUQuestion]
save_jsonl(records: list[dict], path: Path) -> None
```

### Entradas

* Diretório Spider.
* Arquivos MMLU.
* Config YAML.

### Saídas

* Objetos de domínio.
* Arquivos processados.

---

## 9.2 `src/tp2/prompts.py`

**Responsabilidade:** gerar prompts determinísticos.

### Funções esperadas

```python
build_spider_prompt(
    example: SpiderExample,
    schema_text: str,
    few_shot_examples: list[SpiderExample],
) -> str

build_mmlu_prompt(
    question: MMLUQuestion,
    few_shot_examples: list[MMLUQuestion],
) -> str
```

### Regras

* Não variar template por modelo.
* Não incluir dev no few-shot Spider.
* Sempre pedir “responda apenas com SQL” no Spider.
* Sempre pedir uma alternativa `A`, `B`, `C` ou `D` no MMLU.

---

## 9.3 `src/tp2/model.py`

**Responsabilidade:** carregar modelo, tokenizer e adapters.

### Funções esperadas

```python
load_tokenizer(model_name_or_path: str)
load_base_model(config: ModelConfig)
load_model_with_adapter(base_model_name: str, adapter_path: Path, config: ModelConfig)
build_lora_config(config: TrainingConfig)
```

### Regras

* O modelo base deve ser igual para baseline, Exp A e Exp B.
* Adapters devem ser carregados explicitamente.
* Perfil L4 e T4 devem ser controlados por config.

A configuração LoRA deve usar `LoraConfig` com parâmetros como `r`, `target_modules`, `lora_alpha` e `lora_dropout`, que são campos previstos na API do PEFT. ([Hugging Face][6])

---

## 9.4 `src/tp2/inference.py`

**Responsabilidade:** geração e parsing.

### Funções esperadas

```python
generate_text(model, tokenizer, prompt: str, generation_config: dict) -> str
extract_sql(raw_output: str) -> str
extract_mmlu_answer(raw_output: str) -> str | None
```

### Regras para `extract_sql`

A função deve remover:

* blocos ```sql;
* blocos ``` genéricos;
* texto antes do primeiro `SELECT` ou `WITH`;
* texto depois do ponto-e-vírgula final, se houver;
* explicações como “Here is the SQL”.

### Regras para `extract_mmlu_answer`

Ordem de parsing:

1. Match exato de resposta isolada: `A`, `B`, `C`, `D`.
2. Match em padrões como `Answer: A`.
3. Primeira letra válida se a saída for curta.
4. Caso contrário, retornar `None`.

---

## 9.5 `src/tp2/sql_utils.py`

**Responsabilidade:** executar SQL com segurança e comparar resultados.

### Funções esperadas

```python
is_safe_select_query(sql: str) -> bool
execute_sql(db_path: Path, sql: str, timeout_seconds: int = 5) -> QueryResult
normalize_result(rows: list[tuple], preserve_order: bool) -> list[tuple]
has_order_by(sql: str) -> bool
compare_sql_results(predicted: QueryResult, gold: QueryResult, preserve_order: bool) -> bool
```

### Regras

* Abrir SQLite em modo read-only quando possível.
* Bloquear comandos destrutivos.
* Aplicar timeout lógico.
* Fechar conexão em `finally`.

---

## 9.6 `custom_metrics/execution_accuracy.py`

**Responsabilidade:** integrar `sql_utils` ao DeepEval.

### Classe esperada

```python
class ExecutionAccuracy(BaseMetric):
    def __init__(self, spider_db_dir: str, threshold: float = 1.0):
        ...

    def measure(self, test_case: LLMTestCase) -> float:
        ...
```

### Observação técnica

O `LLMTestCase` precisa carregar metadados do Spider, principalmente `db_id`. Se o DeepEval não oferecer campo direto suficiente para isso na versão usada, usar `additional_metadata` ou um wrapper interno.

---

## 9.7 `src/tp2/logging_utils.py`

**Responsabilidade:** salvar logs, ambiente e métricas.

### Funções esperadas

```python
setup_logger(name: str, output_dir: Path)
save_environment_snapshot(output_dir: Path)
save_metrics(metrics: dict, path: Path)
append_jsonl(record: dict, path: Path)
write_summary_markdown(experiment_dir: Path)
```

---

# 10. Estrutura sugerida de pastas

A estrutura base fornecida deve ser mantida. Recomendo adicionar `data/`, `outputs/` e `.gitignore`.

```text
.
├── configs/
│   ├── baseline.yaml
│   ├── train_lora_exp_a.yaml
│   ├── train_lora_exp_b.yaml
│   └── eval.yaml
│
├── custom_metrics/
│   ├── __init__.py
│   └── execution_accuracy.py
│
├── scripts/
│   ├── __init__.py
│   ├── prepare_spider.py
│   ├── prepare_mmlu.py
│   ├── train.py
│   ├── evaluate_spider.py
│   ├── evaluate_mmlu.py
│   └── run_benchmarks.py
│
├── src/
│   └── tp2/
│       ├── __init__.py
│       ├── data.py
│       ├── prompts.py
│       ├── model.py
│       ├── sql_utils.py
│       ├── inference.py
│       ├── config.py
│       └── logging_utils.py
│
├── tests/
│   ├── test_execution_accuracy.py
│   ├── test_sql_utils.py
│   ├── test_prompts.py
│   └── test_mmlu_parsing.py
│
├── notebooks/
│   └── colab_runner.ipynb
│
├── data/
│   ├── raw/
│   │   ├── spider/
│   │   └── mmlu/
│   └── processed/
│       ├── spider/
│       └── mmlu_150_suite.json
│
├── outputs/
│   ├── base/
│   ├── exp_a/
│   └── exp_b/
│
├── requirements.txt
├── README.md
├── Makefile
└── .gitignore
```

## `.gitignore` recomendado

```gitignore
__pycache__/
*.pyc
.env
.venv/
.ipynb_checkpoints/

data/raw/
data/processed/
outputs/
models/
.cache/
wandb/

*.log
*.sqlite-shm
*.sqlite-wal
```

---

# 11. Modelo de dados

Este projeto não tem banco de dados próprio. O modelo de dados principal são arquivos JSON/JSONL e os bancos SQLite do Spider.

## 11.1 Entidade `SpiderExample`

```json
{
  "example_id": "spider-dev-000001",
  "db_id": "concert_singer",
  "question": "What are the names of all singers?",
  "gold_sql": "SELECT name FROM singer",
  "split": "dev"
}
```

| Campo        |   Tipo | Obrigatório | Descrição                     |
| ------------ | -----: | ----------: | ----------------------------- |
| `example_id` | string |         sim | ID interno determinístico     |
| `db_id`      | string |         sim | Identificador do banco Spider |
| `question`   | string |         sim | Pergunta em linguagem natural |
| `gold_sql`   | string |         sim | SQL de referência             |
| `split`      | string |         sim | `train` ou `dev`              |

## 11.2 Entidade `SpiderSchema`

```json
{
  "db_id": "concert_singer",
  "tables": [
    {
      "name": "singer",
      "columns": [
        {"name": "singer_id", "type": "number", "primary_key": true},
        {"name": "name", "type": "text", "primary_key": false}
      ]
    }
  ],
  "foreign_keys": [
    {
      "from_table": "concert",
      "from_column": "singer_id",
      "to_table": "singer",
      "to_column": "singer_id"
    }
  ]
}
```

## 11.3 Entidade `SpiderPrediction`

````json
{
  "example_id": "spider-dev-000001",
  "db_id": "concert_singer",
  "question": "...",
  "prompt_hash": "sha256:...",
  "gold_sql": "SELECT name FROM singer",
  "raw_output": "```sql\nSELECT name FROM singer;\n```",
  "predicted_sql": "SELECT name FROM singer;",
  "score": 1.0,
  "error_type": null,
  "error_message": null,
  "latency_seconds": 1.24
}
````

### `error_type` permitido

```text
null
sql_extraction_error
unsafe_sql
sqlite_connection_error
execution_error
gold_execution_error
result_mismatch
timeout
unknown_error
```

## 11.4 Entidade `MMLUQuestion`

```json
{
  "question_id": "mmlu-stem-000001",
  "category": "STEM",
  "subcategory": "college_computer_science",
  "question": "...",
  "choices": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "answer": "C"
}
```

## 11.5 Entidade `MMLUPrediction`

```json
{
  "question_id": "mmlu-stem-000001",
  "category": "STEM",
  "subcategory": "college_computer_science",
  "raw_output": "C",
  "parsed_answer": "C",
  "gold_answer": "C",
  "is_correct": true,
  "error_type": null
}
```

## 11.6 Entidade `ExperimentSummary`

```json
{
  "experiment_name": "exp_a",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "adapter_path": "outputs/exp_a/adapter",
  "spider_execution_accuracy": 0.0,
  "mmlu_accuracy": 0.0,
  "mmlu_by_category": {
    "STEM": 0.0,
    "Humanidades": 0.0,
    "Sociais": 0.0
  },
  "delta_vs_base": {
    "spider": 0.0,
    "mmlu": 0.0,
    "mmlu_percent": 0.0
  }
}
```

---

# 12. APIs, endpoints ou interfaces

Não há API HTTP no escopo. As interfaces públicas do projeto são comandos CLI e funções Python internas.

## 12.1 CLI — `prepare_spider`

```bash
python -m scripts.prepare_spider \
  --data_dir data/raw/spider \
  --output_dir data/processed/spider
```

### Argumentos

| Argumento      | Obrigatório | Descrição                  |
| -------------- | ----------: | -------------------------- |
| `--data_dir`   |         sim | Diretório com Spider bruto |
| `--output_dir` |         não | Diretório processado       |

### Saídas

* `data/processed/spider/train.jsonl`
* `data/processed/spider/dev.jsonl`
* `data/processed/spider/schemas.json`

---

## 12.2 CLI — `prepare_mmlu`

```bash
python -m scripts.prepare_mmlu --config configs/eval.yaml
```

### Saída

```text
data/processed/mmlu_150_suite.json
```

---

## 12.3 CLI — `train`

```bash
python -m scripts.train --config configs/train_lora_exp_a.yaml
```

### Contrato

| Entrada             | Saída                                  |
| ------------------- | -------------------------------------- |
| YAML de treinamento | Adapter, tokenizer, logs, config final |

### Códigos de erro recomendados

| Código | Situação                   |
| -----: | -------------------------- |
|    `1` | Config inválida            |
|    `2` | Dataset ausente            |
|    `3` | Modelo não carregou        |
|    `4` | OOM ou erro de treinamento |
|    `5` | Falha ao salvar artefatos  |

---

## 12.4 CLI — `evaluate_spider`

```bash
python -m scripts.evaluate_spider \
  --config configs/eval.yaml \
  --model_path outputs/exp_a \
  --output_dir outputs/exp_a
```

### Saídas

* `spider_predictions.jsonl`
* `spider_metrics.json`

---

## 12.5 CLI — `evaluate_mmlu`

```bash
python -m scripts.evaluate_mmlu \
  --config configs/eval.yaml \
  --model_path outputs/exp_a \
  --output_dir outputs/exp_a
```

### Saídas

* `mmlu_predictions.jsonl`
* `mmlu_metrics.json`

---

## 12.6 CLI — `run_benchmarks`

```bash
python -m scripts.run_benchmarks \
  --config configs/eval.yaml \
  --model_path outputs/exp_a
```

### Responsabilidade

Executar:

1. Spider dev.
2. MMLU 150.
3. Cálculo de deltas contra baseline.
4. Escrita de `summary.md`.

---

# 13. Integrações externas

## 13.1 Hugging Face Hub

**Uso:**

* Baixar modelo base.
* Baixar dataset MMLU.
* Carregar tokenizer.

**Falhas possíveis:**

| Falha               | Estratégia                                |
| ------------------- | ----------------------------------------- |
| Sem internet        | Usar cache local                          |
| Rate limit          | Autenticar com token local                |
| Modelo indisponível | Encerrar com erro claro                   |
| Revisão mudou       | Registrar revision/commit quando possível |

## 13.2 TRL

**Uso:** treinamento supervisionado com `SFTTrainer`.

**Justificativa:** reduz código próprio de treinamento e integra bem com modelos Hugging Face. O `SFTTrainer` aceita formatos padrão e conversacionais, úteis para formatar Spider como prompt-completion. ([Hugging Face][4])

## 13.3 PEFT

**Uso:** LoRA e QLoRA.

**Justificativa:** treina poucos parâmetros adicionais e mantém o modelo base congelado. A documentação de integração TRL + PEFT descreve suporte a LoRA e QLoRA. ([Hugging Face][5])

## 13.4 BitsAndBytes

**Uso:** fallback T4 com QLoRA 4-bit.

**Configuração recomendada para T4:**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

A documentação do PEFT mostra uso de `BitsAndBytesConfig` com `load_in_4bit=True`, `nf4`, double quantization e compute dtype para treinar modelos quantizados com LoRA. ([Hugging Face][7])

## 13.5 SQLite

**Uso:**

* Executar SQL prevista.
* Executar SQL gold.
* Comparar resultados.

**Estratégias de segurança:**

* Abrir banco em modo read-only.
* Bloquear comandos não-SELECT.
* Usar timeout.
* Fechar conexão sempre.

## 13.6 Google Colab

**Uso:**

* Ambiente principal de execução.
* Perfil L4 como preferencial.
* Perfil T4 como fallback.

**Regra:** o notebook apenas executa comandos do repositório.

---

# 14. Tratamento de erros

## 14.1 Padrão de erro

Todo erro salvo em JSONL deve seguir:

```json
{
  "error_type": "execution_error",
  "error_message": "no such column: singer_name"
}
```

## 14.2 Erros de configuração

| Erro                    | Exemplo                       | Ação     |
| ----------------------- | ----------------------------- | -------- |
| `config_file_not_found` | YAML inexistente              | Encerrar |
| `invalid_config_schema` | Campo obrigatório ausente     | Encerrar |
| `invalid_profile`       | Perfil diferente de `l4`/`t4` | Encerrar |

## 14.3 Erros de dados

| Erro                     | Ação                                 |
| ------------------------ | ------------------------------------ |
| `spider_train_not_found` | Encerrar preparação                  |
| `spider_dev_not_found`   | Encerrar avaliação                   |
| `sqlite_db_not_found`    | Marcar exemplos do `db_id` como erro |
| `invalid_mmlu_suite`     | Encerrar avaliação MMLU              |

## 14.4 Erros de inferência

| Erro               | Ação                                 |
| ------------------ | ------------------------------------ |
| OOM                | Reduzir batch via config e registrar |
| Timeout de geração | Marcar exemplo como erro             |
| Output vazio       | Marcar parsing como erro             |
| Adapter ausente    | Encerrar com erro claro              |

## 14.5 Erros SQL

| Erro                | Score | `error_type`           |
| ------------------- | ----: | ---------------------- |
| SQL não extraída    | `0.0` | `sql_extraction_error` |
| SQL destrutiva      | `0.0` | `unsafe_sql`           |
| Sintaxe inválida    | `0.0` | `execution_error`      |
| Coluna inexistente  | `0.0` | `execution_error`      |
| Tabela inexistente  | `0.0` | `execution_error`      |
| Resultado diferente | `0.0` | `result_mismatch`      |

---

# 15. Segurança

## 15.1 Execução SQL segura

A SQL vem de um modelo generativo. Portanto, deve ser tratada como entrada não confiável.

### Regras

* Permitir apenas SQL iniciando com `SELECT` ou `WITH`.
* Bloquear termos:

  * `DROP`
  * `DELETE`
  * `UPDATE`
  * `INSERT`
  * `ALTER`
  * `ATTACH`
  * `DETACH`
  * `PRAGMA`
  * `VACUUM`
  * `CREATE`
  * `REPLACE`
* Executar em banco read-only.
* Preferir cópia temporária dos bancos Spider.
* Definir timeout.

## 15.2 Segredos

* Não versionar token Hugging Face.
* Usar variável de ambiente `HF_TOKEN`, se necessário.
* `.env` deve ficar no `.gitignore`.

## 15.3 Dados

* Spider e MMLU são datasets de benchmark.
* Não há dados pessoais no fluxo esperado.
* Mesmo assim, logs não devem incluir tokens, paths privados ou credenciais.

## 15.4 Abuso de recursos

* Permitir `--limit` para smoke test.
* Controlar batch size.
* Encerrar processo de forma limpa em OOM.

---

# 16. Observabilidade e logs

## 16.1 Arquivos de log

```text
outputs/<exp>/
├── train_logs.json
├── environment.json
├── spider_predictions.jsonl
├── spider_metrics.json
├── mmlu_predictions.jsonl
├── mmlu_metrics.json
└── summary.md
```

## 16.2 `environment.json`

```json
{
  "run_id": "2026-06-09T22-00-00_exp_a",
  "python_version": "3.x",
  "torch_version": "...",
  "transformers_version": "...",
  "trl_version": "...",
  "peft_version": "...",
  "deepeval_version": "...",
  "cuda_available": true,
  "gpu_name": "NVIDIA L4",
  "gpu_vram_gb": 24,
  "seed": 42,
  "model_name": "Qwen/Qwen2.5-3B-Instruct"
}
```

## 16.3 Métricas obrigatórias

### Spider

```json
{
  "execution_accuracy": 0.0,
  "total": 0,
  "correct": 0,
  "failed": 0,
  "error_breakdown": {
    "execution_error": 0,
    "result_mismatch": 0,
    "sql_extraction_error": 0
  }
}
```

### MMLU

```json
{
  "accuracy": 0.0,
  "total": 150,
  "correct": 0,
  "by_category": {
    "STEM": {"total": 50, "correct": 0, "accuracy": 0.0},
    "Humanidades": {"total": 50, "correct": 0, "accuracy": 0.0},
    "Sociais": {"total": 50, "correct": 0, "accuracy": 0.0}
  }
}
```

## 16.4 Eventos que devem ser registrados

| Evento                            | Nível      |
| --------------------------------- | ---------- |
| Início/fim de preparação de dados | INFO       |
| Início/fim de treinamento         | INFO       |
| Config efetiva carregada          | INFO       |
| GPU detectada                     | INFO       |
| Erro de SQL por exemplo           | DEBUG/INFO |
| OOM                               | ERROR      |
| Falha de parsing MMLU             | INFO       |
| Métricas finais                   | INFO       |

---

# 17. Estratégia de testes

## 17.1 Testes unitários

### `test_sql_utils.py`

Cenários mínimos:

* `SELECT` válido permitido.
* `DROP TABLE` bloqueado.
* `UPDATE` bloqueado.
* Detecção de `ORDER BY`.
* Normalização ignora ordem quando não há `ORDER BY`.
* Normalização preserva ordem quando há `ORDER BY`.

### `test_execution_accuracy.py`

Cenários mínimos:

| Caso                                   |                  Esperado |
| -------------------------------------- | ------------------------: |
| SQL prevista igual à gold              |                     `1.0` |
| SQL equivalente com texto diferente    |                     `1.0` |
| SQL com ordem diferente sem `ORDER BY` |                     `1.0` |
| SQL com ordem diferente com `ORDER BY` |                     `0.0` |
| Coluna inexistente                     |                     `0.0` |
| Markdown com bloco SQL                 | SQL extraída corretamente |
| Texto explicativo antes da SQL         | SQL extraída corretamente |

### `test_mmlu_parsing.py`

Cenários mínimos:

| Saída                         | Esperado                    |
| ----------------------------- | --------------------------- |
| `A`                           | `A`                         |
| `Answer: B`                   | `B`                         |
| `The correct answer is C.`    | `C`                         |
| `I think it is D`             | `D`                         |
| saída longa com várias letras | `None` ou regra documentada |

## 17.2 Testes de integração

Criar SQLite artificial em memória ou arquivo temporário:

```sql
CREATE TABLE singer (
  id INTEGER PRIMARY KEY,
  name TEXT
);

INSERT INTO singer VALUES (1, 'Ana'), (2, 'Bruno');
```

Validar:

* Execução gold.
* Execução predicted.
* Comparação equivalente.
* Erro de coluna inexistente.

## 17.3 Smoke tests

Adicionar comandos:

```bash
make test
make smoke-spider
make smoke-mmlu
```

Smoke test deve rodar:

* 3 exemplos Spider.
* 6 questões MMLU.
* Sem treinamento completo.

## 17.4 Teste end-to-end mínimo

Fluxo:

1. Preparar dados.
2. Rodar baseline com `--limit 2`.
3. Gerar JSONL.
4. Validar schema dos outputs.
5. Validar que métricas existem.

---

# 18. Critérios de aceite

## 18.1 Repositório

* [ ] Estrutura de pastas segue a documentação.
* [ ] `README.md` explica instalação, preparação, treino e avaliação.
* [ ] `requirements.txt` contém versões fixadas.
* [ ] Não há paths absolutos no código.
* [ ] Notebook Colab apenas executa comandos.

## 18.2 Métrica

* [ ] `ExecutionAccuracy` herda de `BaseMetric`.
* [ ] `measure()` retorna `float`.
* [ ] SQL é extraída de markdown e texto extra.
* [ ] SQL prevista e gold são executadas no SQLite correto.
* [ ] Erros retornam `0.0`.
* [ ] Ordem é tratada conforme `ORDER BY`.
* [ ] Testes unitários passam.

## 18.3 Treinamento

* [ ] `train.py` roda por YAML.
* [ ] Seeds são fixadas.
* [ ] Modelo e tokenizer carregam.
* [ ] LoRA/QLoRA é aplicado.
* [ ] Adapter é salvo.
* [ ] Logs são salvos.
* [ ] Exp A e Exp B rodam com configs distintas.

## 18.4 Avaliação

* [ ] Baseline Spider gera JSONL e métricas.
* [ ] Baseline MMLU gera JSONL e métricas.
* [ ] Exp A gera métricas Spider e MMLU.
* [ ] Exp B gera métricas Spider e MMLU.
* [ ] `run_benchmarks.py` calcula variações contra baseline.
* [ ] `summary.md` contém tabela comparativa.

## 18.5 Relatório

* [ ] Explica pipeline de dados.
* [ ] Mostra configuração LoRA.
* [ ] Explica arquitetura da métrica.
* [ ] Reporta Spider Execution Accuracy.
* [ ] Reporta MMLU geral e por categoria.
* [ ] Analisa 2–3 falhas do modelo fine-tuned.
* [ ] Discute especialização vs generalização.
* [ ] Discute contaminação de dados.
* [ ] Tem até 10 páginas em formato IEEE/ACM.

---

# 19. Roadmap técnico de implementação

## Fase 1 — Base do repositório

1. Criar estrutura de pastas.
2. Criar `requirements.txt`.
3. Criar `config.py` com loader YAML.
4. Criar `logging_utils.py`.
5. Criar `Makefile`.

**Valor entregue:** projeto executável com base organizada.

---

## Fase 2 — Métrica e SQL

1. Implementar `sql_utils.py`.
2. Implementar `extract_sql`.
3. Implementar `ExecutionAccuracy`.
4. Criar banco SQLite artificial nos testes.
5. Cobrir casos com e sem `ORDER BY`.

**Valor entregue:** avaliação Text-to-SQL confiável antes de treinar.

---

## Fase 3 — Preparação de dados

1. Implementar `prepare_spider.py`.
2. Implementar serialização de schemas.
3. Implementar seleção few-shot Spider.
4. Implementar `prepare_mmlu.py`.
5. Persistir `mmlu_150_suite.json`.

**Valor entregue:** dados fixos e auditáveis.

---

## Fase 4 — Baseline

1. Implementar `model.py`.
2. Implementar `inference.py`.
3. Implementar `evaluate_spider.py`.
4. Implementar `evaluate_mmlu.py`.
5. Rodar baseline completo.

**Valor entregue:** números base para comparação.

---

## Fase 5 — Fine-tuning

1. Implementar formatação SFT.
2. Implementar LoRA profile L4.
3. Implementar QLoRA profile T4.
4. Rodar Exp A.
5. Rodar Exp B.
6. Salvar adapters e logs.

**Valor entregue:** modelos especializados.

---

## Fase 6 — Benchmarks finais

1. Implementar `run_benchmarks.py`.
2. Avaliar Exp A.
3. Avaliar Exp B.
4. Calcular deltas.
5. Gerar `summary.md`.

**Valor entregue:** resultados finais comparáveis.

---

## Fase 7 — Relatório

1. Montar tabela principal.
2. Selecionar 2–3 erros qualitativos.
3. Discutir trade-off.
4. Discutir contaminação.
5. Validar reprodutibilidade com README.

**Valor entregue:** entrega acadêmica completa.

---

# 20. Riscos técnicos e decisões pendentes

## 20.1 Riscos

| Risco                           | Impacto                     | Mitigação                                                        |
| ------------------------------- | --------------------------- | ---------------------------------------------------------------- |
| OOM na T4                       | Treino falha                | Usar QLoRA, batch 1, grad accumulation                           |
| Prompt muito longo              | Truncamento prejudica SQL   | Controlar `max_seq_length` e serializar schema de forma compacta |
| SQL gerada com explicação       | Métrica falha indevidamente | Implementar extração robusta                                     |
| SQLite executa comando inseguro | Risco operacional           | Bloquear comandos não-SELECT                                     |
| MMLU parsing ambíguo            | Acurácia incorreta          | Regra rígida de parsing                                          |
| Resultados variam               | Perda de reprodutibilidade  | Seeds, greedy decoding, suite fixa                               |
| Baseline já conhece Spider/MMLU | Interpretação enviesada     | Discutir contaminação                                            |
| Diferença de versões CUDA/libs  | Execução falha no Colab     | Fixar versões após validação                                     |
| DeepEval API muda               | Métrica quebra              | Pin de versão no `requirements.txt`                              |
| Avaliação Spider lenta          | Tempo excessivo             | Permitir `--limit` e logs incrementais                           |

## 20.2 Decisões pendentes

| Decisão                              | Recomendação                                                        |
| ------------------------------------ | ------------------------------------------------------------------- |
| Versões exatas do `requirements.txt` | Validar em Colab L4 e congelar com `pip freeze`                     |
| Subcategorias finais MMLU            | Usar `college_computer_science`, `philosophy`, `econometrics`       |
| `max_seq_length` final L4            | Começar com 4096; reduzir se houver OOM                             |
| `max_seq_length` final T4            | Começar com 2048 ou 3072                                            |
| Uso de Flash Attention               | Opcional; não depender disso para reprodutibilidade                 |
| Uso de W&B/MLflow                    | Fora do caminho crítico; logs locais bastam                         |
| Amostra parcial do Spider dev        | Avaliação final deve usar dev completo, salvo limitação documentada |

---

# 21. Glossário técnico e de negócio

| Termo                   | Definição                                                                      |
| ----------------------- | ------------------------------------------------------------------------------ |
| LLM                     | Modelo de linguagem de grande porte.                                           |
| Fine-tuning             | Ajuste de um modelo pré-treinado em dados específicos.                         |
| Text-to-SQL             | Tarefa de converter pergunta em linguagem natural para SQL.                    |
| Spider                  | Benchmark cross-domain de Text-to-SQL com bancos SQLite, perguntas e SQL gold. |
| Schema linking          | Capacidade de associar termos da pergunta a tabelas e colunas do banco.        |
| SQL gold                | Consulta SQL de referência do dataset.                                         |
| Execution Accuracy      | Métrica que compara o resultado executado da SQL prevista com o da SQL gold.   |
| Baseline                | Modelo base sem fine-tuning.                                                   |
| Adapter                 | Pesos adicionais treinados por LoRA/QLoRA.                                     |
| LoRA                    | Técnica PEFT que treina matrizes adicionais de baixo rank.                     |
| QLoRA                   | LoRA aplicada sobre modelo quantizado, normalmente em 4 bits.                  |
| PEFT                    | Fine-tuning eficiente em parâmetros.                                           |
| TRL                     | Biblioteca Hugging Face para treinamento de modelos de linguagem.              |
| MMLU                    | Benchmark de múltipla escolha para medir conhecimento geral.                   |
| 5-shot                  | Prompt com 5 exemplos resolvidos antes da questão avaliada.                    |
| Greedy decoding         | Decodificação determinística que escolhe o token mais provável.                |
| Temperatura 0           | Configuração de geração determinística.                                        |
| Catastrophic forgetting | Perda de capacidade geral após especialização.                                 |
| Regressão de capacidade | Queda de desempenho em tarefas fora do domínio treinado.                       |
| Contaminação de dados   | Possibilidade de o modelo já ter visto dados do benchmark no pré-treino.       |
| JSONL                   | Formato com um objeto JSON por linha.                                          |
| OOM                     | Out of Memory; erro por falta de memória GPU/RAM.                              |

---

# Configurações recomendadas

## `configs/train_lora_exp_a.yaml`

```yaml
experiment_name: exp_a
seed: 42

paths:
  data_dir: data/processed/spider
  output_dir: outputs/exp_a
  model_cache_dir: .cache/huggingface

model:
  name: Qwen/Qwen2.5-3B-Instruct
  profile: l4
  attn_implementation: sdpa

training:
  method: lora
  learning_rate: 1.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 4096
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_strategy: epoch
  bf16: true
  fp16: false
  gradient_checkpointing: false

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
  bias: none
  task_type: CAUSAL_LM

generation_eval:
  temperature: 0.0
  do_sample: false
  max_new_tokens: 256
```

## `configs/train_lora_exp_b.yaml`

```yaml
experiment_name: exp_b
seed: 42

paths:
  data_dir: data/processed/spider
  output_dir: outputs/exp_b
  model_cache_dir: .cache/huggingface

model:
  name: Qwen/Qwen2.5-3B-Instruct
  profile: l4
  attn_implementation: sdpa

training:
  method: lora
  learning_rate: 2.0e-4
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 4096
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_strategy: epoch
  bf16: true
  fp16: false
  gradient_checkpointing: false

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
  bias: none
  task_type: CAUSAL_LM

generation_eval:
  temperature: 0.0
  do_sample: false
  max_new_tokens: 256
```

## Perfil T4 fallback

```yaml
model:
  name: Qwen/Qwen2.5-3B-Instruct
  profile: t4
  quantization: qlora_4bit

training:
  method: qlora
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  bf16: false
  fp16: true
  gradient_checkpointing: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: float16
```

## `configs/eval.yaml`

```yaml
seed: 42

paths:
  spider_data_dir: data/processed/spider
  spider_db_dir: data/raw/spider/database
  mmlu_suite_path: data/processed/mmlu_150_suite.json
  baseline_dir: outputs/base

model:
  name: Qwen/Qwen2.5-3B-Instruct
  profile: l4

spider:
  split: dev
  few_shot_count: 3
  max_new_tokens: 256

mmlu:
  total_questions: 150
  few_shot_count: 5
  categories:
    STEM:
      subcategory: college_computer_science
      count: 50
    Humanidades:
      subcategory: philosophy
      count: 50
    Sociais:
      subcategory: econometrics
      count: 50

generation:
  temperature: 0.0
  do_sample: false
  max_new_tokens: 256

execution:
  sqlite_timeout_seconds: 5
  allow_only_select: true
```

---

# Resumo para os desenvolvedores

Implementem primeiro a **métrica `ExecutionAccuracy`** e os testes de SQL. Esse é o coração do projeto. Sem uma métrica confiável, qualquer resultado de fine-tuning perde valor.

Depois implementem a preparação dos dados, o prompt fixo do Spider, a suíte MMLU 150 e o baseline. Só treinem os adapters depois que o baseline gerar `spider_predictions.jsonl`, `spider_metrics.json`, `mmlu_predictions.jsonl` e `mmlu_metrics.json`.

A arquitetura deve ser um **modular monolith com scripts CLI**. A lógica principal fica em `src/tp2/`. O Colab apenas clona o repositório, instala dependências e executa comandos.

A ordem recomendada é:

1. `sql_utils.py`
2. `custom_metrics/execution_accuracy.py`
3. testes unitários da métrica
4. `prepare_spider.py`
5. `prepare_mmlu.py`
6. `prompts.py`
7. `evaluate_spider.py`
8. `evaluate_mmlu.py`
9. baseline completo
10. `train.py`
11. Exp A
12. Exp B
13. `run_benchmarks.py`
14. `summary.md`
15. relatório final

As decisões que ainda precisam ser confirmadas antes da execução final são: versões exatas do `requirements.txt`, subcategorias definitivas do MMLU, `max_seq_length` que cabe no Colab usado e se a execução final será no perfil L4 ou T4.

[1]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct?utm_source=chatgpt.com "Qwen/Qwen2.5-3B-Instruct"
[2]: https://huggingface.co/datasets/cais/mmlu?utm_source=chatgpt.com "cais/mmlu · Datasets at Hugging Face"
[3]: https://deepeval.com/docs/metrics-custom?utm_source=chatgpt.com "'Do it yourself' Metrics | DeepEval - The LLM Evaluation ..."
[4]: https://huggingface.co/docs/trl/en/sft_trainer "SFT Trainer · Hugging Face"
[5]: https://huggingface.co/docs/trl/en/peft_integration "PEFT Integration · Hugging Face"
[6]: https://huggingface.co/docs/peft/package_reference/lora "LoRA · Hugging Face"
[7]: https://huggingface.co/docs/peft/en/developer_guides/quantization "Quantization · Hugging Face"
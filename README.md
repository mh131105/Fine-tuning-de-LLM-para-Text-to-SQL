# TP2 NLP - Text-to-SQL Fine-Tuning & Evaluation Pipeline

## 1. Visão Geral e Objetivo
Este projeto implementa um pipeline completo e reprodutível para avaliar a viabilidade de realizar o *fine-tuning supervisionado* de um modelo pequeno com foco em programação (**Qwen/Qwen2.5-Coder-3B-Instruct**) para a tarefa avançada de Text-to-SQL (usando a base **Spider**). O objetivo central é medir se os ganhos de performance nessa tarefa de domínio específico geram "catastrophic forgetting" ou regressões nas suas capacidades genéricas de conhecimento, testadas via **MMLU**.

## 2. Arquitetura do Sistema
O projeto é estruturado em uma pipeline de dados e avaliação composta por 5 componentes principais:

- **Camada de Configuração (`configs/`)**: Arquivos YAML que parametrizam toda a execução (dados, hiperparâmetros do modelo, treino e logs de avaliação).
- **Camada de Scripts (`scripts/`)**: Orquestram os fluxos do começo ao fim (preparação do dataset, fine-tuning via QLoRA, avaliação Spider/MMLU e agregação de dados).
- **Core (`src/`)**: Fornece módulos reutilizáveis, como o carregamento seguro do LLM com quantização de 4-bits, rotinas de geração determinística, construção de prompts e serialização do schema do SQLite.
- **Métricas Customizadas (`custom_metrics/`)**: Implementa a `ExecutionAccuracyMetric` (baseada no framework *DeepEval*). A métrica faz parsing do output gerado, ignorando markdown extra, e roda a SQL predita *read-only* contra o banco de dados SQLite oficial, pontuando matches reais ao invés de usar similaridade semântica insegura.
- **Testes (`tests/`)**: Garante, por meio do `pytest`, que o parsing de schemas, cálculos de acurácia, extratores de SQL e travas de reprodutibilidade estejam puras e asserindo os formatos corretamente.

### Fluxo de Processamento
1. **Preparo**: Conversão dos schemas do SQLite via JSON do Spider -> Construção de Chat Messages TRL -> Exportação final `spider_train_sft.jsonl`.
2. **Treino**: `train_qlora.py` aplica as otimizações PEFT/QLoRA sobre o modelo quantizado.
3. **Avaliação Spider**: Geração inferencial no Spider Dev; extração rigorosa de sintaxe SQL e execução virtual local no SQLite.
4. **Avaliação MMLU**: Geração e parsing A/B/C/D sobre o dataset de conhecimentos gerais (baseline e fine-tuned adapters).
5. **Relatório**: Agregação dos deltas comparativos em tabelas finais `.csv` e `.md`.

## 3. Plano Experimental
Os experimentos foram modelados para aferir as variações de proficiência do modelo sob diferentes cargas e hiperparâmetros PEFT:

- **Baseline**: Qwen2.5-Coder-3B-Instruct avaliado "vanilla" (sem peso alterado). 3-shot prompt com DDL no Spider, e 5-shot prompt no MMLU.
- **Experimento A (QLoRA Focado)**: Fine-tuning apenas nos módulos de atenção principais (`q_proj`, `v_proj`).
  - *Parâmetros*: `r=16`, `lora_alpha=16`, `learning_rate=2e-4`.
  - *Hipótese*: Aprendizado eficiente da relação coluna/entidade sem afetar pesos latentes base.
- **Experimento B (QLoRA Amplo)**: Fine-tuning em maior número de camadas (atenção e feed-forward, como `gate_proj`, `up_proj`, `down_proj`).
  - *Parâmetros*: `r=32`, `lora_alpha=32`, `learning_rate=1e-4`.
  - *Hipótese*: Aumento expressivo em Text-to-SQL, mas maior risco de esquecimento e perda de capacidade generalista refletida em queda na MMLU Accuracy.

As **métricas principais** extraídas e avaliadas cruzando essas 3 fases são: *Execution Accuracy* (acerto de banco relacional), *Invalid SQL Rate* (syntax errors da IA), e *MMLU Accuracy* (impacto cognitivo genérico).

## 4. Garantias de Reprodutibilidade
O projeto e seus scripts são empacotados visando robustez científica sem viés ou variância invisível:

- **Variáveis Determinísticas e Sementes**: Modificações centralizadas sob YAML. Sementes para inicialização `torch`, `numpy.random` e `random` são travadas e injetadas sistemicamente pelo módulo `src/reproducibility.py`.
- **Identificação com Hashes**: As rotinas emitem e comparam meta-dados de run sob chaves MD5/SHA256 (`config_hash`) das flags aplicadas.
- **Ambiente Tracker**: A captação do OS, pacote GPU, memórias VRAM, CUDA e locks das bibliotecas ficam retidas nos outputs das métricas.
- **Execução Real**: Testes empíricos (SQL Execution) ignoram alinhamentos visuais e forçam computação idêntica ao banco esperado.
- **Ponto de Partida**: Um caderno único `notebooks/00_setup_colab.ipynb` agrupa e consolida desde instalações a downloads em Colab.

Para utilizar, instale os componentes de `requirements.txt` e inicie o pipeline pelo Jupyter Notebook base de provisionamento.

# Arquitetura do Sistema

O projeto é estruturado em uma pipeline de dados e avaliação composta por 5 componentes principais:

1. **Camada de Configuração (`configs/`)**: YAMLs que controlam toda a execução (dados, modelo, treino, avaliação).
2. **Camada de Scripts (`scripts/`)**: Orquestram os processos (preparação de dataset, treinamento QLoRA, avaliação Spider/MMLU e agregação).
3. **Core (`src/`)**: Fornece funções reusáveis como carregamento seguro do modelo com 4-bit, geração determinística de SQL, serialização de schema e prompts.
4. **Métricas (`custom_metrics/`)**: Implementa `ExecutionAccuracyMetric` baseada no framework DeepEval. Essa métrica executa (read-only) a SQL predita e a SQL Gold contra o banco de dados SQLite real, pontuando 1.0 para matches e 0.0 caso contrário.
5. **Testes (`tests/`)**: Garante que funções críticas (extratores de SQL, métrica de acurácia, reprodutibilidade de sementes e serialização de schemas) sejam puras e assertivas.

## Fluxo de Processamento

1. **Preparo**: Spider JSON -> SQLite Schema -> TRL Chat Messages -> `spider_train_sft.jsonl`.
2. **Treino**: `train_qlora.py` aplica quantização 4-bit + adaptadores PEFT.
3. **Avaliação Spider**: Geração sobre a base Dev, SQL é extraído, executado localmente.
4. **Avaliação MMLU**: Geração e parsing A/B/C/D em subcategorias variadas.
5. **Relatório**: Os resultados geram `.csv` e `.md` finais.

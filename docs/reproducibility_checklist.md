# Checklist de Reprodutibilidade

Para que os experimentos tenham validade rigorosa, todas as etapas incorporam mecanismos sistêmicos de reprodutibilidade:

- [x] **Configurações Centralizadas**: Nenhuma variável mágica está solta no código. Tudo é orquestrado por YAMLs no diretório `configs/`.
- [x] **Identificação de Experimentos**: Os scripts de preparação e treino geram hashes de configuração (`config_hash`) para identificar exclusivamente aquele run.
- [x] **Sementes Fixas Globalmente**: `src/reproducibility.py` fixa a seed em módulos como `random`, `numpy.random` e funções do `torch`, incluindo `cudnn.deterministic`.
- [x] **Metadados de Ambiente**: Coleta de sistema operacional, versão do Python, pacote Torch e dados da GPU, salvos em `outputs/metrics/environment.json`.
- [x] **Acurácia Baseada em Execução**: Textos gerados são convertidos estritamente em SQL e rodados diretamente em banco real (`sqlite3`), removendo métricas enganosas baseadas puramente em semântica de string (Exact Match).
- [x] **Lock de Dependências**: Todas as bibliotecas fixadas em `requirements.txt`.
- [x] **Prompt Versionado**: A estrutura do prompt (texto e shots) foi fixada via template hashing em `src/prompts.py` e extração isolada na avaliação.

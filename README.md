# TP2 NLP - Text-to-SQL com Qwen2.5-Coder

Este projeto implementa uma pipeline completa para fine-tuning e avaliação do modelo Qwen/Qwen2.5-Coder-3B-Instruct na tarefa de Text-to-SQL, utilizando a base de dados Spider. Inclui também métricas customizadas de Execution Accuracy via DeepEval e regressão de capacidades utilizando o dataset MMLU.

## Estrutura do Projeto

- `configs/`: Arquivos YAML de configuração parametrizada.
- `src/`: Lógica central reutilizável (prompts, schemas, model loader, SQL utils).
- `scripts/`: Scripts orquestradores (preparação, treinamento, avaliação).
- `notebooks/`: Notebooks de experimentação (setup Colab).
- `custom_metrics/`: Métrica Execution Accuracy implementada sob o padrão DeepEval.
- `tests/`: Testes unitários validando extração de SQL, schemas e reprodutibilidade.
- `outputs/`: Diretório gerado contendo métricas, resultados e modelos fine-tuned.

## Execução

1. Instalar as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Preparar os dados Spider:
   ```bash
   python scripts/prepare_spider.py --config configs/data.yaml --split train
   python scripts/prepare_spider.py --config configs/data.yaml --split dev
   ```

3. Avaliar Baseline:
   ```bash
   python scripts/run_baseline_spider.py --config configs/eval_spider.yaml
   ```

4. Realizar o Fine-tuning (QLoRA):
   ```bash
   python scripts/train_qlora.py --config configs/train_exp_a.yaml
   ```

5. Avaliar Fine-Tuned e MMLU e gerar relatórios:
   ```bash
   python scripts/evaluate_spider.py --config configs/eval_spider.yaml --adapter_path outputs/finetuned_exp_a/adapters
   python scripts/prepare_mmlu.py --config configs/eval_mmlu.yaml
   python scripts/aggregate_results.py
   ```

## Licença
Distribuído sob a licença MIT.

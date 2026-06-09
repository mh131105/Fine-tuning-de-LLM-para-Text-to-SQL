# Plano Experimental

## Objetivo Geral
Avaliar a viabilidade de realizar o fine-tuning supervisionado de um modelo pequeno com foco em programação (Qwen2.5-Coder-3B-Instruct) para a tarefa avançada de Text-to-SQL (base Spider), e medir se esse ganho em domínio reduz suas capacidades genéricas de conhecimento (regression em MMLU).

## Experimentos

### 1. Baseline
- **Descrição**: Avaliar o Qwen2.5-Coder-3B base sem nenhum ajuste de pesos.
- **Avaliações**:
  - Spider: Prompt Few-Shot com 3 exemplos fixos contendo DDL.
  - MMLU: Prompt 5-shot por categoria.

### 2. Experimento A (QLoRA Focado)
- **Descrição**: Fine-tuning apenas nos módulos de atenção principais (`q_proj`, `v_proj`).
- **Parâmetros**: `r=16`, `lora_alpha=16`, `learning_rate=2e-4`.
- **Hipótese**: O modelo aprende o formato e a relação entre colunas de forma eficiente sem perturbar o conhecimento base.

### 3. Experimento B (QLoRA Amplo)
- **Descrição**: Fine-tuning em mais módulos (attention e mlp, como `gate_proj`, `up_proj`, `down_proj`).
- **Parâmetros**: `r=32`, `lora_alpha=32`, `learning_rate=1e-4`.
- **Hipótese**: O modelo aumenta expressivamente a capacidade em Text-to-SQL, mas tem maior propensão a esquecimento catastrófico (catastrophic forgetting), o que se refletirá no MMLU.

## Métricas Coletadas
1. **Execution Accuracy**: Se a query retornou o mesmo resultado real no SQLite.
2. **Invalid SQL Rate**: Frequência em que o modelo gerou SQL malformado.
3. **MMLU Accuracy**: Percentual de acerto no MMLU (sinal de regressão de capacidade geral).

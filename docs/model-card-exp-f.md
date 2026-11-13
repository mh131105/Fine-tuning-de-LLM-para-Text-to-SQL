---
base_model: Qwen/Qwen2.5-3B-Instruct
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
license: other
license_name: qwen-research
license_link: LICENSE
language:
- en
datasets:
- dreamerdeo/multispider
tags:
- text-to-sql
- sql
- lora
- qwen2
- research
metrics:
- accuracy
---

# Qwen2.5-3B Text-to-SQL — LoRA Exp F

**Built with Qwen.** Adapter LoRA/PEFT do experimento F do meu projeto de Text-to-SQL, desenvolvido no TP2 de Processamento de Linguagem Natural da UFAM (ICC220 / PPGINF528, 2026/01).

Investiguei quanto o fine-tuning no Spider melhora a geração de SQL e como ele afeta uma medida auxiliar de capacidade geral. Escolhi F como o compromisso recomendado no relatório: **68,18% no Spider dev (+8,61 p.p. sobre a base)** e **86/150 na suíte MMLU, igual à base**.

[Relatório final](https://github.com/mh131105/Fine-tuning-de-LLM-para-Text-to-SQL/blob/pipeline/reports/relatorio_final.pdf) · [Código e reprodução](https://github.com/mh131105/Fine-tuning-de-LLM-para-Text-to-SQL) · [Auditoria](https://github.com/mh131105/Fine-tuning-de-LLM-para-Text-to-SQL/blob/pipeline/docs/revisao-tecnica.md)

## Artefato

Publiquei somente o adapter final, sem fazer merge com a base. O carregamento requer `Qwen/Qwen2.5-3B-Instruct`. Os 288 tensores LoRA somam 7.372.800 parâmetros, salvos em float32; BF16 descreve o treino. O SHA-256 de `adapter_model.safetensors` é `fed43c96bde6fe42b0e34d00315534cf75e28e644fa040ca4163eccbdfc8657c`.

Recuperei o artefato de `exp_f-20260612T032941Z-3-001.zip`. Os pesos são idênticos ao checkpoint final `876`, correspondente a duas épocas, e diferentes do checkpoint `438` da primeira época. A configuração de treino coincide com `configs/train_lora_exp_f.yaml` do código. Preservei pesos e configuração PEFT originais; acrescentei documentação, licença e hashes. Os arquivos de tokenizer vieram do checkpoint 438 do mesmo treino, pois o diretório de tokenizer final no arquivo estava vazio. O pipeline não altera o vocabulário.

A configuração original registra `revision: null` para a base. Não há SHA imutável do modelo-base ou dos datasets; o carregamento abaixo usa a revisão disponível da base e não promete reprodução bit a bit.

## Dados e treinamento

Usei 7000 pares do Spider train, obtidos dos arquivos em inglês da fonte `dreamerdeo/multispider`. O preparo registrou 166 schemas e 1034 exemplos dev. O treino recebe esquema e pergunta em texto simples, sem chat template nem exemplos few-shot, e calcula a loss somente na completion SQL com EOS `<|im_end|>`.

| Parâmetro | Exp F |
| --- | --- |
| Método | LoRA sem quantização |
| Módulos | q_proj, k_proj, v_proj, o_proj |
| Rank / alpha / dropout | 16 / 32 / 0,05 |
| Learning rate / épocas | 2e-4 / 2 |
| Microbatch / acumulação / batch efetivo | 2 / 8 / 16 |
| Comprimento máximo | 2048 tokens |
| Otimizador | adamw_torch_fused |
| Warmup / weight decay | 0,03 / 0 |
| Precisão / seed | BF16 / 42 |
| Ambiente | Colab, NVIDIA A100-SXM4-40GB |
| Train loss / runtime | 0,168494 / 1656,37 segundos |

Registrei Python 3.12.13, PyTorch 2.11.0+cu128, Transformers 4.57.6, PEFT 0.19.1, TRL 0.29.1, Datasets 4.8.5, DeepEval 3.9.9 e bitsandbytes 0.49.2. As versões completas e a configuração estão em `environment.json` e `training_config.yaml`. Bitsandbytes estava instalado, mas este experimento não usou QLoRA.

## Avaliação e resultados

Apliquei o mesmo protocolo à base e aos oito adapters A–H. Usei geração greedy (`do_sample=false`), seed 42 e prompts em texto simples.

- **Spider dev:** 1034 exemplos; instrução, esquema, os três primeiros exemplos do train e pergunta final; até 256 tokens, batch de avaliação 16. As stop sequences configuradas cortam o texto após a geração. A métrica customizada extrai SQL e executa previsão e referência no mesmo banco SQLite, somente leitura e timeout de cinco segundos. Aceita SELECT/WITH após filtro. Compara tuplas mantendo ordem das colunas e duplicatas; ignora ordem das linhas salvo se alguma consulta contiver ORDER BY. Arredonda floats a seis casas. Falha de extração/execução ou resultado diferente recebe zero. Este é um protocolo local por execução em uma instância do banco, **não a Test Suite Accuracy oficial do Spider**.
- **MMLU 150:** 50 questões `college_computer_science`, 50 `philosophy` e 50 `econometrics`, amostradas do test de `cais/mmlu`; cinco exemplos dev fixos por subcategoria, batch 64, até quatro tokens, extração de alternativa A–D. Hash das questões: `8fc4621bde8bba56f49b1d9d4e26d50777653a583b46120e82979ce321e8cc6b`. **Não é o benchmark MMLU completo**; cada questão equivale a 0,67 p.p.

| Modelo | Spider acertos | Spider EX local | Delta Spider | MMLU 150 |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen2.5-3B-Instruct | 616/1034 | 59,57% | — | 86/150 (57,33%) |
| **Exp F (este adapter)** | **705/1034** | **68,18%** | **+8,61 p.p.** | **86/150 (57,33%)** |
| Exp G (maior Spider) | 711/1034 | 68,76% | +9,19 p.p. | 85/150 (56,67%) |

Recalculei os agregados de F a partir das predições recuperadas e conferi sua correspondência com o notebook e o relatório. A base e G aparecem como comparações; este repositório de modelo contém somente os pesos F. `spider_metrics.json`, `mmlu_metrics.json` e `benchmark_summary.json` são os originais recuperados. A acurácia MMLU igual à base inclui trocas de questões: não demonstra preservação de todos os acertos.

## Carregar e gerar SQL

Instalação mínima (Python 3.12):

```sh
python -m pip install torch==2.11.0 transformers==4.57.6 peft==0.19.1 accelerate==1.13.0 huggingface_hub==0.36.2 safetensors==0.7.0
```

Exemplo em CUDA compatível com BF16; usa o formato textual de treino, sem few-shot. Ele imprime a saída e não executa consultas:

```python
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "mh131105/qwen2.5-3b-text-to-sql-lora-exp-f"
config = PeftConfig.from_pretrained(adapter_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id).eval()
prompt = (
    "You are a Text-to-SQL assistant. Generate one SQLite query that answers the question. "
    "Use only the provided schema. Return only SQL, with no markdown and no explanation.\n\n"
    "Final task:\nSchema:\nsinger(singer_id, name, country)\n"
    "Question: How many singers are from each country?\nSQL:\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=128,
                            pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Para CPU/MPS e extração da primeira SQL, consulte o [exemplo executável do projeto](https://github.com/mh131105/Fine-tuning-de-LLM-para-Text-to-SQL/blob/pipeline/examples/infer_exp_f.py). Para repetir as métricas, use o pipeline 3-shot e a suíte fixa; esta demonstração zero-shot não é um benchmark.

## Limitações e uso previsto

Este é um artefato de pesquisa e portfólio. Avaliei um modelo-base, uma seed e prompts específicos. Escolhi a configuração a partir de Spider dev, que portanto não é um teste independente após seleção. Não executei uma nova avaliação completa durante a publicação.

A suíte geral cobre só três subcategorias. Benchmarks públicos podem ter participado do pré-treino. A comparação em um banco pode aceitar SQLs com a mesma saída por coincidência e penalizar ordem de colunas. Permanecem erros de schema linking e consultas executáveis semanticamente incorretas. Revisão humana e controles próprios são necessários antes de qualquer uso de SQL gerada em bancos reais; o exemplo não é um executor.

## Licença, fontes e atribuições

O modelo-base usa a [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE), para pesquisa e avaliação não comerciais; uso comercial requer licença da Alibaba Cloud. Incluí a íntegra em `LICENSE` e o aviso exigido em `NOTICE`. O código do projeto tem licença MIT, que não substitui os termos do artefato.

O [Spider](https://yale-lily.github.io/spider) de Tao Yu e colaboradores é distribuído sob CC BY-SA 4.0. A fonte usada foi [dreamerdeo/multispider](https://huggingface.co/datasets/dreamerdeo/multispider), de Longxu Dou e colaboradores; usei os arquivos em inglês. O [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) de Dan Hendrycks e colaboradores, identificado como MIT na fonte, serviu somente à avaliação. Não redistribuo esses corpora neste repositório de modelo. Consulte `NOTICE` para referências, incluindo LoRA, de Edward J. Hu e colaboradores.

O relatório identifica Matheus como autor. Usei Codex nesta revisão editorial, auditoria dos arquivos e preparação da publicação. As fontes originais não discriminam eventual colaboração ou uso de IA na implementação; não atribuo a esta revisão novos treinos ou resultados de benchmark.

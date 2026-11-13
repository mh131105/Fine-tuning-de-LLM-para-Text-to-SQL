# Reprodução e uso do adapter

O [README](../README.md) apresenta os resultados. Este documento reúne os comandos para usar o adapter selecionado, avaliar os mesmos arquivos ou repetir o treinamento. Os comandos devem ser executados na raiz do repositório.

## Ambiente

Os experimentos finais foram executados no Google Colab com NVIDIA A100-SXM4-40GB, Python 3.12.13, PyTorch 2.11.0+cu128, Transformers 4.57.6, PEFT 0.19.1, TRL 0.29.1, Datasets 4.8.5 e DeepEval 3.9.9. As configurações A–H usam LoRA BF16 sem quantização, batch efetivo 16 e seed 42.

O [`requirements.txt`](../requirements.txt) registra as dependências do pipeline, mas mantém o PyTorch do runtime. Em uma máquina nova, instale antes uma versão do [PyTorch compatível com seu ambiente](https://pytorch.org/get-started/locally/). Para reproduzir o ambiente reportado, use Linux/CUDA e uma A100. O template T4/QLoRA é uma alternativa operacional e não corresponde aos experimentos finais.

```sh
git clone --branch pipeline --single-branch \
  https://github.com/mh131105/Fine-tuning-de-LLM-para-Text-to-SQL.git
cd Fine-tuning-de-LLM-para-Text-to-SQL
python -m pip install -r requirements.txt
```

O [runner Colab atual](../notebooks/colab_runner.ipynb) orquestra essas CLIs. O [notebook executado arquivado](../notebooks/archive/colab_runner_executed.ipynb) registra a execução histórica; seus comandos de clone e branch pertencem à época em que o repositório tinha outro nome. Suas saídas foram preservadas.

Use `HF_TOKEN` no ambiente ou nos segredos do Colab se precisar de acesso autenticado ao Hub. Não grave o token em código, configurações ou arquivos versionados.

## Inferência com o adapter publicado

Para apenas carregar a base e o Exp F, instale as dependências menores de inferência:

```sh
python -m pip install -r requirements-inference.txt
python examples/infer_exp_f.py \
  --adapter mh131105/qwen2.5-3b-text-to-sql-lora-exp-f
```

O script seleciona o dispositivo por padrão; `--device cuda`, `--device mps` ou `--device cpu` permite escolhê-lo. `--revision` fixa uma revisão do adapter no Hub. O carregamento exige memória para o modelo-base de 3B, além dos pesos LoRA. O exemplo usa uma pergunta e um esquema pequenos; as acurácias do relatório vêm do protocolo com dados e few-shot descritos em [experimentos](experimentos.md).

O artefato segue a licença Qwen Research, com uso não comercial. **Built with Qwen.** Consulte a [model card](https://huggingface.co/mh131105/qwen2.5-3b-text-to-sql-lora-exp-f) e os detalhes em [publicação](publicacao.md).

## Preparar Spider

### Aquisição automática

```sh
python -m scripts.prepare_spider \
  --data_dir data/raw/spider \
  --output_dir data/processed/spider
```

O script usa `data/raw/spider/` quando encontra os arquivos esperados. Caso contrário, tenta a fonte Hugging Face configurada. A estrutura exigida está em [`data/README.md`](../data/README.md).

### Importar um diretório ou ZIP/TAR local

```sh
python -m scripts.prepare_spider \
  --data_dir data/raw/spider \
  --output_dir data/processed/spider \
  --source_path /caminho/spider.zip
```

Para solicitar a fonte Hugging Face de forma explícita:

```sh
python -m scripts.prepare_spider \
  --data_dir data/raw/spider \
  --output_dir data/processed/spider \
  --source hf --hf_repo dreamerdeo/multispider
```

`--force_download` autoriza substituir a pasta de dados ao importar ou baixar novamente; use-o apenas quando essa substituição for desejada. O preparo gera `train.jsonl`, `dev.jsonl`, `schemas.json` e `metadata.json`. Confira os totais e as listas `missing_databases` e `unreadable_databases` antes de avaliar.

## Restaurar a suíte MMLU final

Para usar as mesmas 150 perguntas e os mesmos exemplos few-shot da execução preservada:

```sh
mkdir -p data/processed
cp reports/evaluation/mmlu_150_suite.json data/processed/mmlu_150_suite.json
```

O caminho corresponde a `paths.mmlu_suite_path` em [`configs/eval.yaml`](../configs/eval.yaml). A suíte inclui 50 questões de cada uma das três subcategorias e cinco exemplos few-shot por subcategoria. O hash em `metadata.question_hash` identifica o conteúdo avaliado; os resultados preservados permitem conferir sua correspondência.

Para criar outra suíte a partir do dataset `cais/mmlu`:

```sh
python -m scripts.prepare_mmlu --config configs/eval.yaml
```

Esse comando escreve no mesmo caminho. Preserve a suíte anterior antes de executá-lo. A seed controla a amostragem, mas a reprodução exata também depende da versão e da ordem dos registros obtidos da fonte; por isso o arquivo persistido está disponível no repositório.

## Avaliar o baseline e o Exp F sem retreinar

Prepare Spider e restaure a suíte antes desses comandos. O benchmark da base escreve em `outputs/base/` e fornece a referência para os deltas:

```sh
python -m scripts.run_benchmarks \
  --config configs/eval.yaml --model_path outputs/base
```

Baixe o adapter publicado para o layout reconhecido pelo carregador:

```sh
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mh131105/qwen2.5-3b-text-to-sql-lora-exp-f",
    revision="43cd94a450708bf4d9878bf3dff56145eb5bce2c",
    local_dir="outputs/exp_f/adapter",
    allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
)
PY
```

A chamada fixa a revisão publicada do adapter, registrada em [publicação](publicacao.md). A revisão imutável da base usada no treino original não foi registrada. Confira que o download contém a configuração e os pesos antes de avaliar:

```sh
python - <<'PY'
from pathlib import Path

adapter = Path("outputs/exp_f/adapter")
for name in ("adapter_config.json", "adapter_model.safetensors"):
    path = adapter / name
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f"Arquivo do adapter ausente ou vazio: {path}")
print("Configuração e pesos presentes.")
PY
python -m scripts.run_benchmarks \
  --config configs/eval.yaml --model_path outputs/exp_f
```

Essa conferência evita um detalhe do código atual: quando não encontra `adapter_config.json`, o carregador avalia o modelo-base. `--model_path` deve apontar para a pasta do experimento, para `adapter/` ou para um checkpoint local reconhecido. O benchmark não baixa um adapter apenas por receber seu identificador do Hub.

Os comandos completos avaliam **1034 exemplos Spider e 150 questões MMLU**, sem `--limit` e sem `--mock`. A configuração usa batch 16 em Spider e 64 em MMLU; reduza esses valores em uma cópia da configuração se precisar de menos memória. Registre a alteração e preserve o mesmo protocolo e a mesma suíte.

## Repetir o treinamento

Para repetir a configuração escolhida:

```sh
python -m scripts.train --config configs/train_lora_exp_f.yaml
python -m scripts.run_benchmarks \
  --config configs/eval.yaml --model_path outputs/exp_f
```

O treino usa Spider train, loss apenas na completion, EOS `<|im_end|>`, rank 16, alpha 32, dropout 0,05, módulos `q/k/v/o`, learning rate `2e-4`, duas épocas, sequência máxima 2048 e batch efetivo 16. Não usa o Spider dev para atualizar pesos. A comparação das configurações no dev orientou a seleção de F; esse conjunto não é um teste independente após a seleção.

As configurações A–H seguem o padrão `configs/train_lora_exp_<letra>.yaml`. Para repetir G, por exemplo:

```sh
python -m scripts.train --config configs/train_lora_exp_g.yaml
python -m scripts.run_benchmarks \
  --config configs/eval.yaml --model_path outputs/exp_g
```

No Makefile, `train-all` executa A–D, `train-extra` executa E/F e `train-3epoch` executa G/H. Os alvos `bench-all`, `bench-extra` e `bench-3epoch` seguem os mesmos grupos; `bench-all` inclui também o baseline. Os nomes refletem os grupos históricos, não um único alvo com todos os oito treinos.

Antes de repetir um experimento, preserve e mova a pasta `outputs/<exp>/` anterior. Não misture pesos, configurações e métricas de execuções diferentes. O template QLoRA/T4 e o Exp I de quatro épocas não integram a comparação final A–H.

## Verificar o código sem baixar o Qwen

```sh
python -m pytest
```

Os testes usam exemplos pequenos e mocks para verificar preparação, prompts, extração de SQL, comparação por execução e encadeamento das CLIs. Eles não revalidam as acurácias do relatório.

`--dry_run` em `scripts.train` valida a formatação dos dados sem carregar o modelo. `--mock` nos avaliadores usa respostas gold e serve para smoke tests. Como esses comandos também escrevem arquivos, execute-os com configurações e pastas separadas da suíte e dos resultados reais. Não interprete a acurácia de um mock como desempenho do modelo.

## Preservar e conferir uma execução

Guarde o adapter e o tokenizer, `training_config.yaml`, `train_logs.json`, `dataset_preview.json`, `environment.json`, as predições JSONL, os arquivos de métricas e os summaries descritos em [`outputs/README.md`](../outputs/README.md). Preserve também a suíte MMLU, a revisão do código e a identificação dos dados Spider usados.

Os avaliadores sobrescrevem os arquivos na pasta de saída. Salvar somente o notebook executado não guarda os pesos LoRA nem os bancos SQLite. As evidências públicas selecionadas ficam em [`reports/evaluation/`](../reports/evaluation/); a [revisão técnica](revisao-tecnica.md) registra o alcance da conferência atual.

O resultado de execução aqui depende da extração de SQL, do SQLite e da comparação implementada no projeto. Ele não é a Test Suite Accuracy oficial do Spider. A suíte MMLU cobre três subcategorias; um empate agregado com a base não comprova preservação de capacidade geral fora desse recorte.

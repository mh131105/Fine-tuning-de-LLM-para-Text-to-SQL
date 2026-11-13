# Publicação do Exp F

[Adapter no Hugging Face](https://huggingface.co/mh131105/qwen2.5-3b-text-to-sql-lora-exp-f) · [Model card versionada](model-card-exp-f.md)

Escolhi publicar o adapter LoRA do Exp F, recomendado na seção 6.1 do relatório. Ele corresponde a duas épocas, LR 2e-4, batch efetivo 16, rank 16, alpha 32 e dropout 0,05 nos módulos q/k/v/o do `Qwen/Qwen2.5-3B-Instruct`. Não fiz novo treino nem merge com a base.

## Correspondência do arquivo

Recuperei `exp_f/adapter/adapter_model.safetensors` do backup `exp_f-20260612T032941Z-3-001.zip`, cujo SHA-256 é `4aa6398ed628ce2b8fe50f2eff2140faff0e81534a95f5d61c64b9745ee58cbd`. Os pesos têm 29.529.752 bytes e SHA-256:

```text
fed43c96bde6fe42b0e34d00315534cf75e28e644fa040ca4163eccbdfc8657c
```

Eles coincidem com o checkpoint final 876 e diferem do checkpoint 438. A configuração salva coincide com o YAML F versionado. Métricas e predições do mesmo diretório confirmam 705/1034 em Spider e 86/150 em MMLU. Os tensores LoRA são float32; a configuração de treinamento usa BF16.

O diretório final de tokenizer do backup estava vazio. Para disponibilizar o tokenizer, recuperei os seis arquivos do checkpoint 438 do mesmo treino. O treinamento não adiciona tokens ao vocabulário. Mantive o `adapter_config.json` original, inclusive `revision: null`; a revisão imutável da base não foi registrada.

## Conteúdo no Hub

- Pesos e configuração PEFT na raiz, para carregamento com `PeftModel.from_pretrained`.
- Tokenizer, configuração de treino e avaliação, ambiente e logs originais.
- Métricas agregadas originais de F e manifesto SHA-256 dos arquivos publicados.
- Model card, licença Qwen Research completa e `NOTICE` com atribuições.

Os metadados usam `base_model`, `base_model_relation: adapter`, `library_name: peft`, `pipeline_tag: text-generation`, idioma inglês e fonte de treino `dreamerdeo/multispider`. A licença é `other`, com `license_name: qwen-research` e `license_link: LICENSE`, conforme o formato aceito pelo Hub. Não rotulei a suíte parcial como resultado do benchmark MMLU completo nos metadados de leaderboard.

Não publiquei estados de otimizador, pickles de argumentos, checkpoints intermediários ou credenciais. A autenticação usa a CLI do Hugging Face autenticada pelo autor, fora dos arquivos versionados.

## Licenças e referências

**Built with Qwen.** A [Qwen Research License do modelo-base](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) permite pesquisa e avaliação não comerciais; uso comercial requer licença da Alibaba Cloud. A redistribuição exige cópia dos termos, aviso das modificações, atribuição em `NOTICE` e a identificação “Built with Qwen”. Incluí esses materiais no Hub. A [licença MIT do código](../LICENSE) não substitui os termos do modelo.

O [Spider](https://yale-lily.github.io/spider), de Tao Yu e colaboradores, distribui seus dados sob CC BY-SA 4.0; o [repositório de código Spider](https://github.com/taoyds/spider) declara Apache-2.0. São materiais distintos. O projeto usou os arquivos Spider em inglês da fonte [dreamerdeo/multispider](https://huggingface.co/datasets/dreamerdeo/multispider), cuja card identifica licença genérica `cc`; mantive a atribuição e os termos específicos de Spider para as evidências derivadas.

O [cais/mmlu](https://huggingface.co/datasets/cais/mmlu), de Dan Hendrycks e colaboradores, declara MIT. Ele serviu à avaliação, sem participação no fine-tuning. As [evidências versionadas](../reports/evaluation/README.md) incluem os avisos e termos dos dados. O repositório de modelo contém só métricas agregadas, sem redistribuição dos corpora.

Referências adicionais: [Spider, Yu et al.](https://aclanthology.org/D18-1425/), [MultiSpider, Dou et al.](https://arxiv.org/abs/2212.13492), [MMLU, Hendrycks et al.](https://arxiv.org/abs/2009.03300), [LoRA, Hu et al.](https://arxiv.org/abs/2106.09685), [família Qwen](https://arxiv.org/abs/2407.10671) e [formato PEFT](https://huggingface.co/docs/peft/en/tutorial/peft_model_config).

## Verificação da publicação

Publiquei os arquivos na revisão [`7ed0a9d2531a74c3d63e35f1e273ee1199889e26`](https://huggingface.co/mh131105/qwen2.5-3b-text-to-sql-lora-exp-f/commit/7ed0a9d2531a74c3d63e35f1e273ee1199889e26).

Baixei a revisão publicada sem autenticação e conferi tamanho e SHA-256 dos 18 arquivos listados no manifesto, incluindo os pesos originais. O Hub contém também o próprio manifesto e o `.gitattributes` gerado pela plataforma. Validei o carregamento da configuração PEFT, os 288 tensores (7.372.800 parâmetros) e os metadados da model card. O [registro de integridade](../reports/evaluation/hub-verification.json) preserva essa conferência.

Em 13 de novembro de 2026, carreguei a revisão publicada com PEFT e executei o [exemplo de inferência](../examples/infer_exp_f.py) em MPS/float16. Para `How many singers are there?`, no esquema `singer(singer_id, name, country)`, o modelo gerou:

```sql
SELECT count(*) FROM singer
```

Executei essa consulta em um SQLite de teste com três registros e obtive `[(3,)]`, o resultado esperado. O [registro do teste](../reports/evaluation/hub-inference-test.json) conserva as versões, a revisão do adapter e a revisão da base usada **nesta verificação** (`aa8e72537993ba99e69dfaafa59ed015b17504d1`). Essa revisão da base não pode ser atribuída retroativamente ao treino original, que não a registrou.

Esse teste zero-shot verifica download, carregamento e inferência; não substitui os benchmarks originais. Os 51 testes leves do repositório também passaram. O auditor de evidências conferiu os 40 arquivos originais recuperados, a suíte e os agregados E–H e do notebook.

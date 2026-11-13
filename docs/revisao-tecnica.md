# Revisão técnica da apresentação e das evidências

Revisão realizada em 13 de novembro de 2026, a partir da branch padrão `pipeline`, confirmada no remoto e no GitHub. O ponto inicial foi o commit `b4959327c6abd6ee76c5d11d5e40eb3abe11db9f`. A revisão começou pela leitura integral das cinco páginas do [relatório final](../reports/relatorio_final.pdf), incluindo tabelas, figuras, discussão e ameaças à validade.

## Fontes examinadas

| Fonte | O que permite conferir |
| --- | --- |
| [Relatório final](../reports/relatorio_final.pdf) | Objetivo, oito experimentos, comparação de resultados, decisão F/G e limitações originais. |
| [Notebook executado](../notebooks/archive/colab_runner_executed.ipynb) | Saídas originais de preparo, treino e benchmark, com os agregados finais da base e de A–H. |
| [Resumo extraído do notebook](../reports/evaluation/notebook_summary.json) | Células de origem e resultados estruturados, com SHA-256 do notebook. |
| [Evidências E/F/G/H](../reports/evaluation/) | Predições por exemplo, métricas, configuração, logs e ambiente dos backups que correspondem à comparação final. |
| [Suíte MMLU](../reports/evaluation/mmlu_150_suite.json) | As 150 perguntas, os gabaritos e os exemplos few-shot preservados. |
| [Configuração de F](../configs/train_lora_exp_f.yaml) e [avaliação](../configs/eval.yaml) | Hiperparâmetros, dados, limites de geração e condições de execução. |
| [Código da métrica](../custom_metrics/execution_accuracy.py), [SQL](../src/sql_utils.py), [avaliação](../src/evaluation.py) e [prompts](../src/prompts.py) | Definição efetiva das métricas e dos prompts. |

Os nomes dos backups identificam sua origem, sem atribuir a eles a condição de execução final apenas pela data. Os arquivos `exp_e-20260612T032941Z-3-001.zip`, `exp_f-20260612T032941Z-3-001.zip`, `exp_g-20260612T053227Z-3-001.zip` e `exp_h-20260612T053227Z-3-001.zip` contêm as evidências E–H que conferem com o relatório e o notebook.

## Correspondência dos resultados

A tabela final do relatório corresponde às saídas de benchmark persistidas: base com 616/1.034 e 86/150; A com 692 e 81 acertos; B com 684 e 83; C com 696 e 84; D com 692 e 85; E com 691 e 84; F com 705 e 86; G com 711 e 85; H com 709 e 84. O primeiro número de cada par se refere ao Spider, o segundo ao MMLU 150.

Para E/F/G/H, a auditoria recalculou as métricas a partir dos scores e rótulos das predições originais. Os totais, acertos, categorias MMLU e tipos de erro Spider conferem com os arquivos de métricas e com o notebook. Os 1.034 IDs de Spider e os 150 de MMLU são únicos em cada experimento. As entradas e os hashes de prompt correspondem entre esses quatro adapters.

A configuração de treinamento salva em F é idêntica a `configs/train_lora_exp_f.yaml`. Seus logs registram 7.000 exemplos, loss 0,16849406165619418 e 1.656,37 segundos de execução. O ambiente registra A100-SXM4-40GB, Python 3.12.13 e as versões de bibliotecas citadas no relatório. O adapter identifica `Qwen/Qwen2.5-3B-Instruct`, PEFT 0.19.1, LoRA rank 16, alpha 32, dropout 0,05 e q/k/v/o. A recomendação de publicar F segue a seção 6.1 do relatório.

## Divergências históricas e sua origem

Os três backups `outputs-20260610T205439Z-3-001.zip`, `outputs-20260611T041906Z-3-001.zip` e `outputs-20260611T135819Z-3-001.zip` conservam resultados de execuções anteriores. Suas próprias métricas conferem com seus scores salvos, mas divergem da comparação final. Por exemplo:

| Backup | Base Spider | A Spider | B Spider | C Spider |
| --- | ---: | ---: | ---: | ---: |
| 10/06, 20:54 UTC | 618 | 460 | 604 | ausente |
| 11/06, 04:19 UTC | 610 | 543 | 613 | 618 |
| 11/06, 13:58 UTC | 616 | 674 | 686 | 669 |
| Execução final no notebook | 616 | 692 | 684 | 696 |

As datas e horas acima vêm dos nomes dos arquivos. Elas identificam backups, não garantem o horário de cada inferência. A origem documental da divergência é a coexistência de snapshots de execuções distintas. Os arquivos disponíveis não permitem atribuir cada diferença a uma alteração específica de código, configuração ou geração.

O caso da base exige atenção: o último backup tem os mesmos 616 acertos, mas separa os erros em 162 `execution_error` e 256 `result_mismatch`. O notebook final e o relatório registram 163 e 255. Comparar suas predições históricas com G produz 158 ganhos e 63 perdas; o relatório registra 157 e 62. Ambos resultam em saldo +95. O total igual encobre diferenças por exemplo. Por essa razão, o baseline histórico não foi incorporado como se fosse o conjunto de predições final.

Arquivos avulsos também pertencem a etapas diferentes: o `spider_predictions.jsonl` recuperado estava vazio; o `mmlu_predictions.jsonl` e o `mmlu_metrics.json` avulsos registravam 39/150, enquanto a suíte avulsa tinha o hash final correto. A auditoria selecionou cada artefato pela correspondência com sua função e com as evidências, preservando a distinção entre configuração, suíte e resultado.

## Alcance da validação

O [auditor de evidências](../scripts/audit_evidence.py) executa com a biblioteca padrão:

```sh
python3 -m scripts.audit_evidence
```

Ele confere tamanhos e hashes dos arquivos com o [manifesto das fontes](../reports/evaluation/source_manifest.json), reextrai os agregados do notebook, confere seu SHA-256 e verifica as métricas E–H contra predições e notebook, IDs e entradas compartilhadas, categorias e gabaritos MMLU, formato e igualdade dos hashes de prompt e o hash de questões da suíte. O manifesto registra hashes dos ZIPs de origem; a execução desse comando verifica os arquivos recuperados, sem exigir os ZIPs no repositório.

O hash de questões usa o algoritmo original de `src/preparation.py`, baseado nos dicionários das 150 questões. Ele depende da serialização original e não cobre os few-shot. A auditoria verifica a presença de cinco exemplos por subcategoria; ela não demonstra, por esse hash, que todos os bytes dos prompts foram os usados na execução original. As predições preservam hashes dos prompts, sem o texto integral de cada prompt.

Esta conferência reaplica a agregação dos scores e a comparação dos rótulos salvos. A auditoria de evidências não repetiu a inferência de benchmark, as SQLs originais ou uma nova avaliação de desempenho. O teste funcional do artefato publicado é documentado em [publicação](publicacao.md). Faltam as predições finais da base e de A–D. Assim, os testes binomiais pareados e as contagens 74/58/18 da seção 3 do relatório permanecem como análises originais, com auditoria independente pendente. Os agregados desses modelos continuam conferíveis no notebook.

## Correções de apresentação e cuidados de reprodução

A apresentação descreve oito experimentos LoRA concluídos. O QLoRA T4 é um template complementar, e I é uma configuração exploratória sem resultado na tabela final. O README anterior dizia dois experimentos, apresentava `reports/` como notas ainda a escrever e incluía comandos de atualização pela branch `main`; essas descrições pertenciam a uma etapa anterior do projeto.

A documentação detalha que a métrica de Spider é própria, usa uma instância SQLite, preserva ordem de colunas e duplicatas e só ignora ordem das linhas sem `ORDER BY`. MMLU 150 é uma suíte de três subcategorias. Igualar sua acurácia agregada à da base não demonstra ausência de regressão em todos os itens, categorias ou capacidades.

O treinamento usa prompt sem few-shot, loss na completion e EOS explícito. A avaliação Spider acrescenta três exemplos fixos de treino. O código passa texto simples ao tokenizer; exemplos de inferência devem respeitar esse formato. As stop sequences são cortes aplicados ao texto após geração. O [documento de experimentos](experimentos.md) registra esses detalhes.

O carregador de avaliação procura `adapter_config.json` em um diretório local, na raiz ou em `adapter/` ou `adapters/`. Se não encontrar, carrega o modelo-base. Para avaliar um adapter do Hub com os CLIs existentes, é necessário baixar seus arquivos para um diretório local antes de usar `--model_path` e conferir a presença da configuração. O ID remoto sozinho não ativa o adapter nesse carregador.

As configurações originais não fixam revisões imutáveis do modelo-base ou dos datasets; o `adapter_config.json` de F registra `revision: null`. Seeds, versões, hashes e arquivos preservados ajudam a reproduzir o protocolo, mas não garantem identidade bit a bit em outro ambiente. A comparação e seleção de configurações no mesmo Spider dev também limitam a interpretação desse resultado como estimativa independente de generalização.

## Autoria e assistência

O relatório identifica Matheus Henrique de Oliveira Garcia como autor. A revisão editorial, a organização dos materiais e a auditoria descritas neste documento tiveram assistência de um agente de IA. As fontes examinadas não oferecem uma divisão detalhada das contribuições anteriores entre pessoas e ferramentas. A documentação não atribui contribuições específicas sem registro que as sustente.

# Evidências da execução final

Este diretório contém os arquivos recuperados que correspondem aos resultados do relatório. Os JSONs, JSONLs e YAMLs de E/F/G/H são cópias dos backups, sem editar seus scores. `source_manifest.json` registra os arquivos de origem e seus hashes. `notebook_summary.json` contém uma extração das saídas originais da base e A–H, com células de origem e hash do notebook.

A suíte `mmlu_150_suite.json` conserva as questões e os exemplos few-shot da avaliação. Seu hash de arquivo é diferente do hash de questões registrado nos metadados; o [auditor](../../scripts/audit_evidence.py) verifica o algoritmo original e os agregados:

```sh
python3 -m scripts.audit_evidence
```

As predições finais da base e A–D não foram recuperadas. A [revisão técnica](../../docs/revisao-tecnica.md) explica os snapshots históricos divergentes e os limites da auditoria.

## Atribuições e licenças dos dados

Os arquivos de predições Spider contêm perguntas e SQLs de referência de **Spider**, de Tao Yu e colaboradores: [artigo EMNLP 2018](https://aclanthology.org/D18-1425/), [fonte e licença](https://yale-lily.github.io/spider). O dataset usa [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), com [termos completos](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en). O pipeline extraiu os arquivos em inglês de [dreamerdeo/multispider](https://huggingface.co/datasets/dreamerdeo/multispider), fonte de [MultiSpider, Dou et al., AAAI 2023](https://arxiv.org/abs/2212.13492). Os registros derivados acrescentam predições, scores e metadados de avaliação; a redação recuperada foi preservada nesta revisão. Esses registros derivados com conteúdo Spider estão disponíveis sob CC BY-SA 4.0.

A suíte e as referências MMLU vêm de **cais/mmlu**, de Dan Hendrycks e colaboradores: [dataset card MIT](https://huggingface.co/datasets/cais/mmlu), [artigo](https://arxiv.org/abs/2009.03300). A [licença MIT da fonte original](MMLU-LICENSE), Copyright (c) 2020 Dan Hendrycks, está incluída neste diretório. A seleção de 150 questões e os rótulos de categoria seguem o preparo do projeto; os exemplos few-shot também estão preservados.

A licença MIT do código na raiz não substitui as licenças dos dados ou do modelo-base. Não há pesos de modelo nem bancos SQLite neste diretório.

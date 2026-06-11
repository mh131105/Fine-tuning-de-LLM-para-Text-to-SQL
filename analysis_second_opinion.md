# Auditoria tecnica dos resultados finais do TP2

Data da auditoria: 2026-06-11.  
Escopo: codigo local, `outputs/`, `outputs/diagnostics/` e especificacao do TP2 em PDF.  
Restricao respeitada: nao rodei treino nem inferencia; usei apenas leitura leve de JSON/JSONL e inspecao de codigo.

## Veredito

A conclusao central e metodologicamente defensavel, com uma formulacao cuidadosa:

> O LoRA em Qwen2.5-3B melhora Spider de 59,57% para ate 66,83% no melhor checkpoint observado, enquanto o MMLU cai de 57,33% para 55,33% na suite de 150 questoes.

Essa frase e defensavel porque os numeros foram recomputados dos JSONL, a avaliacao Spider e pareada, o ganho Spider e estatisticamente forte contra o baseline, e a queda MMLU observada e pequena e nao significativa na suite usada.

Ressalva importante: `66,83%` nao e o `Exp C final`; e o `Exp C checkpoint-875`. O `Exp C final` obteve 64,70%. Portanto, no relatorio, a frase deve dizer "melhor checkpoint observado" ou "checkpoint-875/early stopping". Se o texto apresentar `66,83%` como resultado final do Exp C sem essa ressalva, fica metodologicamente impreciso. Se quiser uma afirmacao mais conservadora e preregistrada, use `Exp B = 66,34%`.

## Criterios da especificacao usados como fonte de rigor

A especificacao exige: Spider train exclusivo para fine-tuning; Spider dev para avaliacao; Execution Accuracy por execucao SQLite como metrica unica de Text-to-SQL; prompt com schema do banco; MMLU com exatamente 150 questoes, 50 por cada uma de 3 categorias; MMLU em 5-shot; LoRA documentado com `r`, `alpha`, `dropout` e `target_modules`; pelo menos duas configuracoes de hiperparametros; sementes fixadas e geracao deterministica.

## Auditoria de codigo

| Area | Evidencia local | Veredito |
| --- | --- | --- |
| Treinamento | `src/training.py` fixa seed, carrega Spider train, formata `prompt` + `completion`, usa `completion_only_loss`, adiciona o EOS de chat do Qwen, cria `SFTTrainer` com LoRA e salva adapter/tokenizer/logs. | Aderente ao desenho do TP2. |
| LoRA | `src/model.py` cria `LoraConfig` com `r`, `lora_alpha`, `lora_dropout`, `target_modules`; `load_for_inference` usa `PeftModel.from_pretrained` quando encontra `adapter_config.json`. | Carregamento LoRA correto para `outputs/<exp>/adapter` e checkpoints. |
| Spider | `src/evaluation.py` carrega Spider dev, usa Spider train para 3-shot, aplica o mesmo prompt, salva `raw_output`, `predicted_sql`, `score` e erros. | Comparabilidade adequada entre baseline e LoRA. |
| Execution Accuracy | `custom_metrics/execution_accuracy.py` extrai SQL, bloqueia SQL destrutivo, executa predicted/gold em SQLite read-only, ignora ordem salvo `ORDER BY`. | Atende ao nucleo exigido. Nao e o avaliador oficial completo do Spider, mas a especificacao permite metrica customizada por execucao. |
| MMLU | `src/evaluation.py` carrega suite persistida, usa 5-shot por subcategoria, greedy/temperature 0, parseia A/B/C/D e salva JSONL. | Aderente; todos os 150 outputs foram parseados. |
| Reprodutibilidade | configs usam seed 42, `temperature: 0.0`, `do_sample: false`, ambiente registra GPU L4 e versoes. | Bom. Caveat: `data/processed/` nao esta presente neste workspace, entao nao consegui verificar diretamente a suite MMLU persistida nem os splits Spider, apenas os outputs. |

## Metricas recomputadas

| Modelo | Spider | Spider acc | MMLU | MMLU acc | Delta Spider vs base | Delta MMLU vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 616/1034 | 59,57% | 86/150 | 57,33% | 0,00 pp | 0,00 pp |
| Exp A | 674/1034 | 65,18% | 82/150 | 54,67% | +5,61 pp | -2,67 pp |
| Exp B | 686/1034 | 66,34% | 83/150 | 55,33% | +6,77 pp | -2,00 pp |
| Exp C final | 669/1034 | 64,70% | 83/150 | 55,33% | +5,13 pp | -2,00 pp |
| Exp C checkpoint-875 | 691/1034 | 66,83% | 83/150 | 55,33% | +7,25 pp | -2,00 pp |

Todos os totais esperados foram confirmados a partir dos JSONL.

### MMLU por categoria

| Modelo | STEM | Humanidades | Sociais |
| --- | ---: | ---: | ---: |
| Base | 32/50 | 28/50 | 26/50 |
| Exp A | 30/50 | 27/50 | 25/50 |
| Exp B | 31/50 | 26/50 | 26/50 |
| Exp C final | 29/50 | 28/50 | 26/50 |
| Exp C checkpoint-875 | 29/50 | 28/50 | 26/50 |

A queda MMLU de 86 para 83 acertos e pequena em valor observado, mas a suite tem apenas 150 itens. IC Wilson 95% aproximado: Base 49,33%-64,97%; Exp B/Exp C 47,34%-63,06%. Os intervalos se sobrepoem muito. O resultado permite dizer "nao houve evidencia de degradacao grande nesta suite", mas nao prova preservacao geral de capacidade.

## McNemar pareado em Spider

Teste exato bicaudal de McNemar, pareado por `example_id`.

| Comparacao | Modelo 1 correto / Modelo 2 errado | Modelo 1 errado / Modelo 2 correto | Delta do Modelo 2 | p exato |
| --- | ---: | ---: | ---: | ---: |
| Base vs Exp A | 73 | 131 | +5,61 pp | 5,92e-05 |
| Base vs Exp B | 73 | 143 | +6,77 pp | 2,20e-06 |
| Base vs Exp C final | 83 | 136 | +5,13 pp | 4,17e-04 |
| Base vs Exp C checkpoint-875 | 62 | 137 | +7,25 pp | 1,11e-07 |
| Exp C final vs checkpoint-875 | 45 | 67 | +2,13 pp para checkpoint | 0,0467 |

O ganho Spider de LoRA contra baseline e robusto para todos os experimentos. A superioridade do checkpoint-875 sobre o Exp C final e menor, mas aparece no limiar de significancia.

## McNemar pareado em MMLU

| Comparacao | Base correto / modelo errado | Base errado / modelo correto | Delta | p exato |
| --- | ---: | ---: | ---: | ---: |
| Base vs Exp A | 4 | 0 | -2,67 pp | 0,125 |
| Base vs Exp B | 4 | 1 | -2,00 pp | 0,375 |
| Base vs Exp C final | 6 | 3 | -2,00 pp | 0,508 |
| Base vs checkpoint-875 | 6 | 3 | -2,00 pp | 0,508 |

Nao ha evidencia estatistica forte de regressao MMLU nesta suite. Isso ajuda a explicar por que a degradacao baixa nao e automaticamente suspeita: a intervencao LoRA e pequena e a avaliacao MMLU e curta.

## Por que Exp C final e pior que checkpoint-875

O Exp C final treinou 2 epocas com `q/k/v/o`, enquanto o checkpoint-875 e o ponto de 1 epoca dentro da execucao de 2 epocas. O treino continuou reduzindo loss, mas Spider dev piorou:

| Artefato | Passos | Epoca | Train loss reportado | Spider |
| --- | ---: | ---: | ---: | ---: |
| Exp C checkpoint-875 | 875 | 1,0 | intermediario, LR ainda ~5,19e-5 no fim da epoca | 691/1034 |
| Exp C final/checkpoint-1750 | 1750 | 2,0 | 0,1669 agregado; logs finais com loss ~0,07-0,10 | 669/1034 |

Isso e compativel com sobreajuste ou com mudanca indesejada de comportamento gerativo. O principal sinal concreto e o aumento de saidas vazias/sem SQL no Exp C final:

| Modelo | `raw_output` vazio | `predicted_sql` vazio | `sql_extraction_error` |
| --- | ---: | ---: | ---: |
| Exp B | 4 | 5 | 5 |
| Exp C final | 50 | 53 | 53 |
| Exp C final sem stop | 50 | 53 | 53 |
| Exp C checkpoint-875 sem stop | 1 | 2 | 2 |

Na comparacao checkpoint-875 vs Exp C final, o checkpoint acerta 67 exemplos que o final erra; em 26 desses, o erro do final e `sql_extraction_error`. O final acerta 45 que o checkpoint erra. Logo, a diferenca liquida de 22 acertos vem em boa parte de falhas de geracao/extração no final.

## Stop sequences

As `stop_sequences` foram refutadas como causa da queda do Exp C final:

| Comparacao | Score diff | Raw output diff | Predicted SQL diff | Resultado |
| --- | ---: | ---: | ---: | --- |
| Exp C final normal vs Exp C final sem stop | 0/1034 | 0/1034 | 0/1034 | Identico |

Se as stop sequences estivessem cortando respostas validas, a avaliacao sem stop deveria produzir `raw_output` ou `score` diferente. Nao produziu.

## Hipotese de early EOS no Exp C final

A hipotese de early EOS e plausivel, mas nao provada conclusivamente pelos artefatos atuais.

Evidencias a favor:

- `Exp C final sem stop` preserva 50 `raw_output` vazios, portanto nao e truncamento por stop sequence.
- O checkpoint-875 do mesmo experimento tem apenas 1 `raw_output` vazio.
- A rotina de inferencia decodifica com `skip_special_tokens=True`; uma geracao que emite imediatamente `<|im_end|>` pode virar string vazia nos JSONL.
- O treinamento adiciona `<|im_end|>` ao fim das completions e a segunda epoca pode ter reforcado demais a probabilidade de parar cedo.

Limite da evidencia:

- Os JSONL nao salvam ids de tokens gerados, token bruto com `skip_special_tokens=False`, `eos_token_id` emitido, nem motivo de parada. Entao nao da para provar que o primeiro token foi EOS sem reexecutar inferencia instrumentada.

Conclusao: early EOS e a melhor explicacao local para parte da queda do Exp C final, mas deve ser escrito como "forte indicio", nao como fato fechado.

## Analise dos hiperparametros

| Fator | Evidencia | Interpretacao |
| --- | --- | --- |
| `q/v` vs `q/k/v/o` | Exp A 65,18%; Exp B 66,34%. | Expandir target modules melhorou Spider em +1,16 pp sem piorar MMLU alem de 1 item em relacao ao A. |
| 1 vs 2 epocas | Exp B 66,34%; Exp C final 64,70%; C checkpoint-875 66,83%. | Segunda epoca nao ajudou; usar checkpoint intermediario e melhor. |
| LR `1e-4` | Todos os experimentos finais usam `1e-4`. | Conservador o suficiente para nao gerar queda MMLU grande; ainda assim 2 epocas geraram falhas de parada. |
| `r=16`, `alpha=32` | Confirmado nos adapters. | Configuracao PEFT moderada; coerente com ganho sem grande regressao geral. |
| `dropout=0.05` | Confirmado nos adapters. | Regularizacao leve; nao impediu a degradacao do Exp C final na segunda epoca. |
| Batch efetivo 8 | `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`. | Conservador para L4 e coerente com estabilidade. |
| `max_seq_length=2048` | Confirmado nos configs. | Suficiente para schema + pergunta; risco de truncamento de treino existe apenas se exemplos muito longos, mas nao ha evidencia nos outputs de que esse seja o erro dominante. |
| `completion_only_loss=true` | Confirmado em config/codigo. | Ajuda a especializar a SQL sem treinar o modelo a copiar prompt; tambem reduz risco de degradacao geral. |

Sobre a duvida "o treinamento foi conservador?": sim para A/B e para o checkpoint-875. O conjunto LoRA, LR `1e-4`, batch efetivo 8, uma epoca, dropout 0,05 e loss apenas na completion e conservador o bastante para preservar MMLU na suite curta. O Exp C final mostra o limite: a segunda epoca deixou de ser conservadora para Spider, mesmo sem grande mudanca no MMLU.

## Hipoteses plausiveis refutadas pelos resultados

| Hipotese | Status | Evidencia |
| --- | --- | --- |
| O ganho Spider e artefato de prompt diferente entre modelos. | Refutada. | `prompt_hash` bate em 1034/1034 exemplos entre base e LoRA. |
| Stop sequences explicam a queda do Exp C final. | Refutada. | Avaliacao com e sem stop gerou outputs e scores identicos. |
| O carregamento LoRA falhou e avaliou modelo base por engano. | Refutada. | Adapters tem configs corretos; resultados diferem do baseline; checkpoints sao carregaveis por `adapter_config.json`. |
| MMLU caiu por erro de parsing. | Refutada. | 150/150 respostas foram parseadas em todos os modelos. |
| A segunda epoca necessariamente melhora Spider. | Refutada. | Checkpoint-875 supera o final de 2 epocas. |
| O target `q/v` ja captura todo o ganho possivel. | Refutada. | `q/k/v/o` melhora de 65,18% para 66,34%/66,83%. |
| Houve esquecimento catastrofico claro em MMLU. | Refutada nesta suite. | Queda de 2 pp, McNemar nao significativo, categorias sem colapso. |
| O checkpoint-875 e pior no MMLU que o final. | Refutada. | Ambos fazem 83/150 com mesmas categorias; predicoes MMLU de Exp C final e checkpoint-875 tem 0 discordancias. |

## Ressalvas metodologicas

1. `66,83%` e resultado de checkpoint intermediario. Deve ser tratado como melhor checkpoint observado ou early stopping, nao como resultado final bruto do Exp C.
2. Spider dev foi usado para comparar e escolher modelos/checkpoints. Isso e aceitavel para o TP2 se reportado de forma transparente, mas nao e uma estimativa imparcial de desempenho futuro apos selecao.
3. `data/processed/` nao esta presente no workspace auditado. Portanto, confirmei os totais e categorias pelos outputs, mas nao revalidei diretamente a construcao da suite MMLU ou a separacao fisica dos splits.
4. A metrica implementada e Execution Accuracy por execucao SQLite, como exigido, mas nao replica integralmente todos os detalhes do avaliador oficial Spider test-suite.
5. MMLU com 150 questoes e pequeno. A frase correta e "queda observada pequena nesta suite", nao "o modelo preserva conhecimento geral de modo amplo".
6. A especificacao pede discutir contaminacao de dados. Os artefatos nao conseguem provar ausencia de contaminacao de Spider/MMLU no pre-treino do Qwen; isso deve ser assumido como limitacao interpretativa.

## Recomendacoes para o relatorio

- Reporte a tabela completa: Base, Exp A, Exp B, Exp C final e Exp C checkpoint-875.
- Use o headline: "melhor checkpoint observado: 66,83% Spider; Exp B planejado: 66,34%; Exp C final: 64,70%".
- Inclua o teste de McNemar para Spider, principalmente Base vs checkpoint-875 (`p=1,11e-07`) e Base vs Exp B (`p=2,20e-06`).
- Para MMLU, diga que a queda de 57,33% para 55,33% equivale a 3 questoes e nao e estatisticamente forte na suite de 150 itens.
- Explique o Exp C final como evidencia contra "mais epocas sempre ajudam": houve menor train loss, mas pior Spider dev e 50 saidas vazias.
- Diga explicitamente que stop sequences foram testadas e refutadas como causa.
- Descreva early EOS como hipotese mais provavel, com a limitacao de que os JSONL nao salvam tokens brutos.
- Se houver tempo para uma revisao futura, adicione validacao separada para escolher checkpoint, salve tokens gerados com `skip_special_tokens=False`, e reporte motivo de parada/EOS.

## Conclusao final

Nao encontrei erro critico que invalide a conclusao de que LoRA melhora Spider com pequena queda observada em MMLU. O resultado e defensavel, mas o texto precisa separar tres afirmacoes:

1. Resultado planejado forte: Exp B melhora Spider de 59,57% para 66,34% com MMLU 55,33%.
2. Melhor resultado observado: Exp C checkpoint-875 chega a 66,83% com MMLU 55,33%.
3. Resultado final de 2 epocas: Exp C final cai para 64,70%, provavelmente por sobre-treinamento/early EOS parcial.

Com essas ressalvas, a narrativa tecnica fica consistente com a especificacao e com os artefatos locais.

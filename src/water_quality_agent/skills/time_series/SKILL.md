---
name: time_series
description: Seleciona e plota séries temporais de parâmetros de qualidade da água, com normalização de unidades e limites CONAMA Classe 2 quando aplicáveis.
---
# Série temporal
Use quando o usuário pedir evolução, histórico ou gráfico temporal.

Procedimento: (1) confirme que `time_series` está disponível em `dataset_capabilities`; (2) resolva o parâmetro com `resolve_water_parameter`; (3) recupere somente o subconjunto necessário com `get_series`; (4) consulte `get_conama_class2_limits`; (5) compare as unidades presentes com a unidade regulatória e chame `normalize_observation_units` apenas se necessário; (6) chame `plot_time_series`, passando mínimo/máximo somente quando existirem. Não execute estatística descritiva ou tendência sem solicitação. Nunca invente conversões ou limites.

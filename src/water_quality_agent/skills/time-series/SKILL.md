---
name: time_series
description: Seleciona e plota séries temporais de parâmetros de qualidade da água, com normalização de unidades e limites CONAMA Classe 2 quando aplicáveis.
---
# Série temporal
Use quando o usuário pedir evolução, histórico ou gráfico temporal.

Procedimento: (1) confirme que `time_series` está disponível em `dataset_capabilities`; (2) resolva o parâmetro com `resolve_water_parameter`; (3) recupere somente o subconjunto necessário com `get_series`; (4) consulte `get_conama_class2_limits`; (5) chame `plot_time_series`, passando mínimo/máximo somente quando existirem. Os dados retornados por `get_series` já estão harmonizados: o parâmetro está padronizado, o resultado está em formato numérico, o qualifier está separado e a unidade já corresponde à unidade analítica final. Não execute estatística descritiva ou tendência sem solicitação. Nunca invente limites.

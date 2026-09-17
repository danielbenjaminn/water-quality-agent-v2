# Water Quality Agent — arquitetura agêntica

Esta versão remove `pipeline.py` como orquestrador. O fluxo fixo foi substituído por: **ingestão semântica validada → sessão de dados → agente LangGraph → skills → tools determinísticas → RAG opcional**.

## Responsabilidades

- `ingestion/`: leitura, profiling, mapeamento semântico, validação obrigatória e relatório de ingestão.
- `core/`: cálculo, parsing, catálogo regulatório e estado do dataset. Não decide o roteiro da análise.
- `tools/`: superfície pequena que o LLM pode chamar. O DataFrame completo não trafega pelo prompt.
- `skills/`: procedimentos em `SKILL.md`; dizem quando e em que ordem usar tools.
- `agent/`: loop LangGraph de tool calling.
- `rag/`: indexação/consulta das referências fornecidas pelo usuário.

## Entrada generalista e bloqueio de segurança

`ingest_dataset()` detecta encoding/separador, cria um perfil compacto das colunas e resolve aliases inequívocos deterministicamente. O LLM recebe apenas nomes, tipos e exemplos para resolver semanticamente as lacunas; ele pode retornar `null` quando não houver evidência suficiente. Latitude/longitude/geolocalização não fazem parte do schema do domínio e permanecem como `unknown`.

São obrigatórios: `date`, `parameter`, `result` e `unit`. `point` é opcional; se ausente, o backend cria um ponto técnico único e registra `point_inferred=True`. Laudo, campanha, classe, matriz, sub-bacia, bacia e qualifier são opcionais. Se qualquer campo obrigatório não for identificado, o dataset não é ativado na sessão e nenhuma análise/tool pode ser executada.

Toda ingestão produz um relatório visível com as colunas encontradas, o significado atribuído (ou `unknown`), os campos obrigatórios presentes/ausentes e a decisão de prosseguir ou bloquear.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
cp .env.example .env
```

Configure a chave do provider e execute:

```bash
python main.py data/dados.csv "Plote a série temporal de OD no ponto P01"
```

## RAG

Os livros/PDFs não são incluídos no repositório. Indexe cópias às quais você tenha acesso:

```bash
python ingest_reference.py /caminho/referencia.pdf
```

A tool `search_technical_references` consulta o índice somente quando a skill/agente precisar fundamentar interpretação ou metodologia.

## O que foi reaproveitado

Os módulos determinísticos de estatística, tendência, correlação, outliers, parsing e matching foram preservados em `core/`. Eles deixaram de ser disparados sequencialmente por uma pipeline e passaram a ser invocados por tools sobre subconjuntos escolhidos pelo agente.

# 💧 Water Quality Agent

Agente para análise de séries históricas de qualidade da água, desenvolvido com **Python, LangChain, LangGraph e Streamlit**.

A aplicação combina um agente LLM com ferramentas determinísticas de análise estatística, avaliação regulatória segundo a **Resolução CONAMA nº 357/2005 — águas doces Classe 2**, visualização de séries temporais e consulta a referências técnicas por meio de RAG.

O objetivo da arquitetura é permitir que o agente escolha, sob demanda, quais análises executar de acordo com a pergunta do usuário, sem depender de uma pipeline analítica fixa.

---

## Visão geral

O fluxo principal da aplicação é:

```text
CSV
 │
 ▼
Ingestão e inspeção semântica
 │
 ▼
Validação do schema
 │
 ▼
Harmonização
 ├── parâmetros
 ├── qualifiers
 └── unidades
 │
 ▼
Dataset canônico em sessão
 │
 ▼
Agente LangGraph
 │
 ├── Tools de dados
 ├── Estatística descritiva
 ├── Tendência temporal
 ├── Correlação de Kendall
 ├── Detecção de outliers
 ├── CONAMA Classe 2
 ├── Avaliação de conformidade
 ├── Séries temporais
 └── RAG de referências técnicas
 │
 ▼
Resposta ao usuário
```

O DataFrame completo permanece no backend. As tools analíticas acessam diretamente o dataset canônico e retornam ao LLM apenas os resultados necessários para interpretação.

---

## Interface

A aplicação possui uma interface de chat desenvolvida em **Streamlit**.

O usuário pode:

1. carregar um arquivo CSV;
2. aguardar a ingestão e harmonização dos dados;
3. fazer perguntas em linguagem natural;
4. solicitar análises estatísticas e regulatórias;
5. solicitar séries temporais;
6. visualizar gráficos diretamente na interface.

Exemplos de perguntas:

```text
Analise o comportamento do oxigênio dissolvido no ponto PV180.

Qual a série temporal de OD no ponto PV180?

Existe tendência temporal significativa para turbidez no ponto PV180?

Qual a conformidade do oxigênio dissolvido com a CONAMA Classe 2?

Calcule a correlação entre oxigênio dissolvido e turbidez no ponto PV180.

Faça uma análise descritiva do fósforo total no ponto AV007.
```

---

## Arquitetura

### `ingestion/`

Responsável pela entrada de datasets heterogêneos.

A ingestão realiza:

- leitura do CSV;
- inspeção das colunas;
- profiling;
- mapeamento semântico;
- validação;
- construção do dataset canônico.

Os campos obrigatórios são:

```text
date
parameter
result
unit
```

O campo `point` é opcional.

Quando ele não existe, o sistema pode tratar o dataset como pertencente a um único ponto lógico e registrar essa informação internamente.

Outros campos contextuais podem ser reconhecidos quando disponíveis.

---

### `core/`

Contém as funções determinísticas do domínio.

Entre as responsabilidades estão:

- preparação dos resultados;
- extração de qualifiers;
- padronização de parâmetros;
- conversão de unidades;
- estatística descritiva;
- correlação;
- tendência temporal;
- outliers;
- limites regulatórios;
- avaliação de conformidade;
- harmonização;
- estado da sessão.

O LLM não executa esses cálculos diretamente.

---

### `tools/`

Expõe ao agente as capacidades executáveis da aplicação.

As principais tools incluem:

#### Dados

- inspeção das capacidades do dataset;
- listagem de pontos;
- listagem de parâmetros;
- resolução de nomes de parâmetros;
- consulta de observações.

#### Análise

- estatística descritiva;
- teste de tendência Mann-Kendall;
- correlação de Kendall;
- identificação de potenciais outliers por IQR.

#### Regulação

- consulta aos limites da CONAMA 357/2005 Classe 2;
- cálculo de conformidade das observações.

#### Visualização

- geração de séries temporais;
- inclusão de limites regulatórios quando aplicáveis.

#### Literatura técnica

- busca semântica em referências indexadas no RAG.

---

## Skills

As skills ficam em:

```text
src/water_quality_agent/skills/
```

Elas descrevem procedimentos metodológicos que orientam o agente sobre como utilizar as tools.

Atualmente existem skills relacionadas a:

```text
descriptive-analysis
time-series
trend-analysis
```

A separação conceitual utilizada no projeto é:

```text
Skill = procedimento/metodologia
Tool  = capacidade executável
Core  = cálculo determinístico
RAG   = conhecimento de referência
```

---

## RAG

O projeto possui uma base vetorial local para consulta de literatura técnica.

A implementação utiliza:

- `sentence-transformers`;
- embeddings locais multilíngues;
- Chroma;
- LangChain;
- PyPDF.

O modelo padrão de embeddings é:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Os embeddings são executados localmente e não dependem do modelo de chat.

### Indexando uma referência

Os PDFs não precisam ser versionados no repositório.

Para indexar uma referência:

```bash
python ingest_reference.py /caminho/para/referencia.pdf
```

Opcionalmente:

```bash
python ingest_reference.py referencia.pdf \
    --title "Título da referência" \
    --author "Autor" \
    --reference-type book
```

O índice é armazenado localmente em:

```text
.vectorstore/
```

A tool:

```text
search_technical_references
```

permite que o agente consulte esse conteúdo quando uma interpretação ou metodologia exigir fundamentação técnica.

---

## CONAMA 357/2005

O projeto trabalha atualmente com **águas doces Classe 2**.

Os limites utilizados pelo sistema ficam em arquivos de configuração e são consultados por tools específicas.

A arquitetura mantém separadas:

```text
dados observacionais
        │
        ├── análise estatística
        │
        └── comparação regulatória
                    │
                    ▼
             CONAMA Classe 2
```

O agente não deve inferir limites legais por conta própria.

A avaliação de conformidade é realizada deterministicamente pelo backend.

---

## Dados censurados

Resultados contendo qualificadores analíticos, como:

```text
<0.05
>10
```

são separados em:

```text
result
qualifier
```

Essa distinção permite que cada metodologia determine como lidar com dados censurados.

Por exemplo, a análise de correlação implementada exclui observações censuradas e utiliza apenas pares efetivamente quantificados.

---

## Correlação

A análise entre parâmetros utiliza **Kendall Tau**.

As observações são pareadas por:

```text
Data + Ponto
```

e somente datas em que os dois parâmetros possuem resultados numéricos simultaneamente participam do cálculo.

A tool recebe os parâmetros solicitados pelo agente e acessa diretamente o dataset armazenado no backend, evitando transportar séries completas pelo contexto do LLM.

---

## Estrutura do projeto

```text
water-quality-agent/
│
├── app.py
├── main.py
├── ingest_reference.py
├── pyproject.toml
├── README.md
├── domain_manifest.yaml
│
├── config/
│   ├── limites_legais.csv
│   ├── parametros_catalogo.csv
│   └── regras_parametros.txt
│
├── data/
│
├── outputs/
│
└── src/
    └── water_quality_agent/
        │
        ├── agent/
        │   ├── graph.py
        │   └── runtime.py
        │
        ├── ingestion/
        │   ├── models.py
        │   ├── profiler.py
        │   ├── semantic_mapper.py
        │   ├── service.py
        │   └── validator.py
        │
        ├── core/
        │   ├── conversao_unidades.py
        │   ├── correlation.py
        │   ├── descriptive.py
        │   ├── domain.py
        │   ├── harmonization.py
        │   ├── outliers.py
        │   ├── parameter_resolution.py
        │   ├── regulatory.py
        │   ├── session.py
        │   └── trend.py
        │
        ├── tools/
        │   ├── analysis_tools.py
        │   ├── data_tools.py
        │   ├── rag_tools.py
        │   ├── regulatory_tools.py
        │   └── visualization_tools.py
        │
        ├── skills/
        │   ├── descriptive-analysis/
        │   ├── time-series/
        │   └── trend-analysis/
        │
        └── rag/
            ├── chunker.py
            ├── embeddings.py
            ├── knowledge_base.py
            ├── loader.py
            ├── metadata.py
            ├── retriever.py
            └── vectorstore.py
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd water-quality-agent
```

### 2. Crie o ambiente virtual

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale o projeto

```bash
pip install -e .
```

---

## Configuração do LLM

Crie um arquivo:

```text
.env
```

na raiz do projeto.

Para OpenAI:

```env
OPENAI_API_KEY=sua_chave
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.6-luna
```

O projeto também possui suporte a Groq no builder de LLM:

```env
GROQ_API_KEY=sua_chave
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
```

Não versione o arquivo `.env`.

---

## Executando a interface

Com o ambiente virtual ativo:

```bash
streamlit run app.py
```

O Streamlit disponibilizará a aplicação localmente, normalmente em:

```text
http://localhost:8501
```

Na interface:

1. selecione um CSV;
2. clique em **Carregar dataset**;
3. aguarde a validação;
4. utilize o chat para solicitar as análises.

---

## Execução por terminal

Também é possível utilizar o agente sem Streamlit:

```bash
python main.py data/dados.csv "Analise o oxigênio dissolvido no ponto P01"
```

O `main.py` realiza a ingestão do dataset e envia a pergunta ao agente.

---

## Princípios de implementação

O projeto foi estruturado em torno de alguns princípios:

### O LLM não é a calculadora

Cálculos estatísticos, conversões, filtros e avaliações regulatórias são realizados por código determinístico.

### O DataFrame permanece no backend

As séries completas não precisam trafegar pelo contexto do modelo para que uma análise seja executada.

### O agente escolhe as capacidades necessárias

Não existe uma sequência fixa de análises executada para toda pergunta.

### Skills e tools possuem responsabilidades diferentes

As skills descrevem procedimentos; as tools executam operações.

### Literatura não substitui cálculo

O RAG é utilizado para fundamentação técnica e metodológica, enquanto resultados quantitativos permanecem responsabilidade das tools determinísticas.

---

## Estado atual

A versão atual implementa:

- ingestão semântica de CSV;
- validação de schema;
- harmonização de parâmetros;
- tratamento de qualifiers;
- conversão de unidades;
- sessão canônica;
- resolução de parâmetros;
- estatística descritiva;
- tendência Mann-Kendall;
- correlação Kendall Tau;
- detecção de outliers por IQR;
- consulta aos limites da CONAMA Classe 2;
- avaliação de conformidade regulatória;
- séries temporais;
- geração de gráficos;
- RAG local de referências técnicas;
- agente LangGraph com tool calling;
- skills metodológicas;
- interface de chat em Streamlit.

---

## Limitações atuais

A aplicação está deliberadamente restrita à **CONAMA 357/2005 — águas doces Classe 2**.

O mapeamento semântico depende da informação disponível no arquivo de entrada. Quando não existe evidência suficiente para identificar uma coluna ou parâmetro de forma segura, a aplicação pode manter o item como não resolvido em vez de assumir uma correspondência.

O tratamento estatístico de dados censurados depende da metodologia utilizada. Nem todas as análises estatísticas para dados censurados estão implementadas.

A interface Streamlit mantém o histórico visual da conversa, mas a versão atual envia ao agente a pergunta corrente em cada execução; portanto, referências conversacionais entre perguntas independentes ainda são limitadas.

---

## Tecnologias

- Python
- pandas
- NumPy
- SciPy
- Matplotlib
- Pydantic
- LangChain
- LangGraph
- OpenAI / Groq
- Chroma
- Sentence Transformers
- PyPDF
- Streamlit

---

## Observação sobre referências

Os documentos técnicos utilizados pelo RAG não são distribuídos junto ao código.

Cada usuário deve indexar apenas documentos aos quais possua acesso legítimo.
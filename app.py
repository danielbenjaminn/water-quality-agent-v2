from __future__ import annotations

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from water_quality_agent.agent.graph import build_agent
from water_quality_agent.ingestion.service import ingest_dataset


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv(".env")

st.set_page_config(
    page_title="Water Quality Agent",
    page_icon="💧",
    layout="centered",
)

st.title("💧 Water Quality Agent")

st.caption(
    "Agente para análise de séries históricas de qualidade da água."
)


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def create_llm():
    return ChatOpenAI(
        model="gpt-5.6-luna",
        temperature=0,
        reasoning_effort="none",
    )


llm = create_llm()


# ============================================================
# AGENTE
# ============================================================

@st.cache_resource
def create_agent():
    return build_agent(llm)


agent = create_agent()


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def extract_generated_images(messages) -> list[str]:
    """
    Procura arquivos PNG retornados pelas tools do agente.

    A tool plot_time_series retorna o caminho do arquivo
    gerado. O Streamlit utiliza esse caminho para renderizar
    a imagem diretamente na conversa.
    """

    images: list[str] = []

    for message in messages:

        if not isinstance(message, ToolMessage):
            continue

        content = message.content

        if not isinstance(content, str):
            continue

        # Remove possíveis aspas adicionadas durante
        # a serialização da resposta da tool.
        path = (
            content
            .strip()
            .strip('"')
            .strip("'")
        )

        if not path.lower().endswith(".png"):
            continue

        if not os.path.exists(path):
            continue

        if path not in images:
            images.append(path)

    return images


# ============================================================
# ESTADO DO STREAMLIT
# ============================================================

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR — DATASET
# ============================================================

with st.sidebar:

    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Selecione um arquivo CSV",
        type=["csv"],
    )

    # --------------------------------------------------------
    # Carregamento
    # --------------------------------------------------------

    if uploaded_file is not None:

        if st.button(
            "Carregar dataset",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Inspecionando e preparando o dataset..."
            ):

                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                # --------------------------------------------
                # A ingestão atual recebe um caminho.
                # O arquivo enviado pelo Streamlit está
                # inicialmente em memória.
                # --------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as tmp:

                    tmp.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = tmp.name

                try:

                    ingestion = ingest_dataset(
                        temp_path,
                        llm=llm,
                    )

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # --------------------------------------------
            # Resultado
            # --------------------------------------------

            if ingestion.get("valid"):

                st.session_state.dataset_loaded = True

                st.session_state.dataset_name = (
                    uploaded_file.name
                )

                # Um novo dataset começa uma nova conversa.
                st.session_state.messages = []

                st.success(
                    "Dataset carregado com sucesso."
                )

            else:

                st.session_state.dataset_loaded = False

                st.error(
                    "Falha na ingestão do dataset."
                )

                with st.expander(
                    "Detalhes da ingestão"
                ):

                    st.json(
                        ingestion
                    )


    # ========================================================
    # STATUS
    # ========================================================

    st.divider()

    if st.session_state.dataset_loaded:

        st.success(
            "Dataset ativo"
        )

        st.write(
            st.session_state.dataset_name
        )

    else:

        st.info(
            "Carregue um CSV para iniciar."
        )


# ============================================================
# SEM DATASET
# ============================================================

if not st.session_state.dataset_loaded:

    st.info(
        "Carregue um arquivo CSV na barra lateral "
        "para começar a análise."
    )


# ============================================================
# HISTÓRICO DA CONVERSA
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        # ----------------------------------------------------
        # Texto
        # ----------------------------------------------------

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Imagens produzidas por tools
        # ----------------------------------------------------

        for image_path in message.get(
            "images",
            [],
        ):

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    use_container_width=True,
                )


# ============================================================
# INPUT DO CHAT
# ============================================================

prompt = st.chat_input(
    "Pergunte sobre os dados...",
    disabled=not st.session_state.dataset_loaded,
)


# ============================================================
# EXECUÇÃO DA PERGUNTA
# ============================================================

if prompt:

    # ========================================================
    # MENSAGEM DO USUÁRIO
    # ========================================================

    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message("user"):

        st.markdown(
            prompt
        )


    # ========================================================
    # RESPOSTA DO AGENTE
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Analisando..."
        ):

            try:

                # --------------------------------------------
                # Por enquanto enviamos somente a pergunta
                # atual ao LangGraph.
                #
                # O histórico visual é mantido pelo Streamlit.
                # Memória conversacional será adicionada
                # separadamente.
                # --------------------------------------------

                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    }
                )

                # --------------------------------------------
                # Mensagens produzidas pelo LangGraph
                # --------------------------------------------

                agent_messages = result.get(
                    "messages",
                    []
                )

                # --------------------------------------------
                # Resposta textual final
                # --------------------------------------------

                if not agent_messages:

                    answer = (
                        "O agente não retornou uma resposta."
                    )

                else:

                    final_message = agent_messages[-1]

                    answer = getattr(
                        final_message,
                        "content",
                        str(final_message),
                    )

                # --------------------------------------------
                # Detecta gráficos produzidos pelas tools
                # --------------------------------------------

                images = extract_generated_images(
                    agent_messages
                )

                # --------------------------------------------
                # Renderiza texto
                # --------------------------------------------

                st.markdown(
                    answer
                )

                # --------------------------------------------
                # Renderiza gráficos
                # --------------------------------------------

                for image_path in images:

                    st.image(
                        image_path,
                        use_container_width=True,
                    )

                # --------------------------------------------
                # Salva resposta no histórico visual
                # --------------------------------------------

                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "images": images,
                }

                st.session_state.messages.append(
                    assistant_message
                )

            except Exception as exc:

                st.error(
                    f"Erro durante a análise: {exc}"
                )
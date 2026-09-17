from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document

class KnowledgeBase:
    """RAG opcional. Indexa referências do usuário; não embute livros protegidos no repositório."""
    def __init__(self, persist_dir: str = ".vectorstore"):
        self.persist_dir = persist_dir
        self.vectorstore = None

    def ingest_pdf(self, path: str, embeddings) -> int:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        docs = PyPDFLoader(path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180).split_documents(docs)
        self.vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=self.persist_dir)
        return len(chunks)

    def open(self, embeddings):
        from langchain_chroma import Chroma
        self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)
        return self

    def search(self, query: str, k: int = 5) -> list[Document]:
        if self.vectorstore is None: return []
        return self.vectorstore.similarity_search(query, k=k)

import argparse
from water_quality_agent.rag.knowledge_base import KnowledgeBase

def main():
    p=argparse.ArgumentParser(); p.add_argument("pdf"); p.add_argument("--provider",default="openai"); args=p.parse_args()
    if args.provider=="openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings=OpenAIEmbeddings()
    else: raise ValueError("Configure um provider de embeddings suportado.")
    n=KnowledgeBase().ingest_pdf(args.pdf,embeddings); print(f"{n} chunks indexados")
if __name__=="__main__": main()

import os

def build_llm():
    provider=os.getenv("LLM_PROVIDER","groq").lower()
    model=os.getenv("LLM_MODEL","openai/gpt-oss-120b")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0)
    raise ValueError(f"LLM_PROVIDER não suportado: {provider}")

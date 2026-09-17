from __future__ import annotations
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from water_quality_agent.skills.registry import SkillRegistry
from water_quality_agent.tools.data_tools import dataset_capabilities, list_parameters, list_points, resolve_water_parameter, get_series
from water_quality_agent.tools.regulatory_tools import get_conama_class2_limits
from water_quality_agent.tools.unit_tools import normalize_observation_units
from water_quality_agent.tools.analysis_tools import descriptive_statistics, mann_kendall_trend, kendall_correlations, iqr_outliers
from water_quality_agent.tools.visualization_tools import plot_time_series
from water_quality_agent.tools.rag_tools import search_technical_references

TOOLS=[dataset_capabilities,list_parameters,list_points,resolve_water_parameter,get_series,get_conama_class2_limits,normalize_observation_units,descriptive_statistics,mann_kendall_trend,kendall_correlations,iqr_outliers,plot_time_series,search_technical_references]

class State(TypedDict): messages: Annotated[list[AnyMessage], add_messages]

def build_agent(llm):
    registry=SkillRegistry()
    model=llm.bind_tools(TOOLS)
    system=SystemMessage(content=f"""Você é um agente de análise de qualidade de águas superficiais doces, restrito neste projeto à CONAMA 357/2005 Classe 2.
O DataFrame fica no backend: use tools para inspecionar/selecionar dados; não peça ao usuário para renomear colunas se o mapeamento semântico já as identificou.
Escolha a metodologia adequada a partir deste catálogo de skills:\n{registry.catalog()}\n
Skills são procedimentos, tools são capacidades executáveis. Não rode análises em massa por padrão. Execute somente as tools necessárias à pergunta. Quando precisar de metodologia detalhada, siga o conteúdo das skills disponibilizadas abaixo:\n\n""" + "\n\n".join(s.instructions for s in registry.skills.values()))
    def call_model(state: State): return {"messages":[model.invoke([system,*state["messages"]])]}
    g=StateGraph(State); g.add_node("agent",call_model); g.add_node("tools",ToolNode(TOOLS)); g.add_edge(START,"agent"); g.add_conditional_edges("agent",tools_condition); g.add_edge("tools","agent")
    return g.compile()

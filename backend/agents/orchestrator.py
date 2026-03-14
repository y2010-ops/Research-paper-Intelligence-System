"""
Multi-agent orchestrator using LangGraph.
Restored 3-step process: Classify -> Search -> Synthesize.
"""
from typing import TypedDict, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

from backend.agents.rag_agent import RAGAgent
from backend.agents.graph_agent import GraphAgent
from backend.storage.vector_store import VectorStore
from backend.storage.knowledge_graph import KnowledgeGraph
from backend.utils.llm_factory import get_llm

class AgentState(TypedDict):
    query: str
    query_type: Literal["factual", "relational", "hybrid"]
    rag_response: Dict
    graph_response: Dict
    final_answer: str

class QueryOrchestrator:
    def __init__(self, vector_store: VectorStore, knowledge_graph: KnowledgeGraph):
        self.rag_agent = RAGAgent(vector_store)
        self.graph_agent = GraphAgent(knowledge_graph)
        self.llm = get_llm(temperature=0.0)
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("classify", self._classify_query)
        workflow.add_node("rag_search", self._run_rag)
        workflow.add_node("graph_search", self._run_graph)
        workflow.add_node("hybrid_search", self._run_hybrid)
        workflow.add_node("synthesize", self._synthesize_answer)
        
        # Add Edges
        workflow.set_entry_point("classify")
        
        workflow.add_conditional_edges(
            "classify",
            lambda x: x["query_type"],
            {
                "factual": "rag_search",
                "relational": "graph_search",
                "hybrid": "hybrid_search"
            }
        )
        
        workflow.add_edge("rag_search", "synthesize")
        workflow.add_edge("graph_search", "synthesize")
        workflow.add_edge("hybrid_search", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()

    async def _classify_query(self, state: AgentState) -> Dict:
        """Step 1: Classify Query"""
        prompt = ChatPromptTemplate.from_template("""
        Classify the query type.
        
        Rules:
        - "factual": Questions asking for definitions, summaries, or content from text (e.g. "What is...", "Explain...").
        - "relational": Questions about authors, citations, or common concepts (e.g. "Who wrote...", "What papers cite...").
        - "hybrid": Complex questions needing both (e.g. "Summarize papers written by...").
        
        Query: {query}
        
        Return ONLY one word: factual, relational, or hybrid.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke({"query": state["query"]})
        classification = result.strip().lower()
        
        # Fallback if LLM gives garbage
        if classification not in ["factual", "relational", "hybrid"]:
            classification = "hybrid"
            
        return {"query_type": classification}

    async def _run_rag(self, state: AgentState) -> Dict:
        """Execute RAG Agent"""
        result = await self.rag_agent.process(state["query"])
        return {"rag_response": result}
        
    async def _run_graph(self, state: AgentState) -> Dict:
        """Execute Graph Agent"""
        result = await self.graph_agent.process(state["query"])
        return {"graph_response": result}
        
    async def _run_hybrid(self, state: AgentState) -> Dict:
        """Execute Both in Parallel"""
        rag_task = asyncio.create_task(self.rag_agent.process(state["query"]))
        graph_task = asyncio.create_task(self.graph_agent.process(state["query"]))
        results = await asyncio.gather(rag_task, graph_task, return_exceptions=True)
        return {
            "rag_response": results[0] if not isinstance(results[0], Exception) else {},
            "graph_response": results[1] if not isinstance(results[1], Exception) else {}
        }

    async def _synthesize_answer(self, state: AgentState) -> Dict:
        """Step 3: Synthesize Final Answer"""
        rag_ans = state.get("rag_response", {}).get("answer", "No text context.")
        graph_ans = state.get("graph_response", {}).get("answer", "No graph context.")
        
        context = f"RAG (Text) Info:\n{rag_ans}\n\nGraph (Relation) Info:\n{graph_ans}"
        
        prompt = ChatPromptTemplate.from_template("""
        You are a research assistant. Synthesize a helpful answer using the provided information.
        
        Query: {query}
        
        Information:
        {context}
        
        Answer (be specific and cite sources if mentioned):
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        final = await chain.ainvoke({"query": state["query"], "context": context})
        
        return {"final_answer": final}

    async def process_query(self, query: str) -> Dict:
        """Run the full workflow"""
        initial_state = {
            "query": query,
            "query_type": "hybrid",
            "rag_response": {},
            "graph_response": {},
            "final_answer": ""
        }
        return await self.workflow.ainvoke(initial_state)
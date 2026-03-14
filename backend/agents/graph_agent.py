"""
Graph Agent for querying Neo4j.
Translates natural language to Cypher using LLM.
"""
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from backend.storage.knowledge_graph import KnowledgeGraph
from backend.utils.llm_factory import get_llm

class GraphAgent:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.llm = get_llm(temperature=0.0)
        
        self.cypher_prompt = ChatPromptTemplate.from_template("""
        You are a Neo4j expert. Convert the question into a Cypher query.
        
        Schema:
        - (Paper)-[:CITES]->(Paper)
        - (Paper)-[:HAS_CONCEPT]->(Concept)
        - (Author)-[:WROTE]->(Paper)
        
        Rules:
        1. Return ONLY the Cypher query. No markdown, no explanations.
        2. Use case-insensitive matching where appropriate (toLower).
        3. Limit results to 10.
        
        Question: {question}
        
        Cypher Query:
        """)
        
        self.chain = self.cypher_prompt | self.llm | StrOutputParser()
        
    async def process(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query against the Knowledge Graph.
        """
        try:
            # 1. Generate Cypher
            cypher_query = await self.chain.ainvoke({"question": query})
            
            # Clean response
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            logger.info(f"Generated Cypher: {cypher_query}")
            
            # 2. Execute Query
            results = self.kg.execute_query(cypher_query)
            
            return {
                "answer": f"Found {len(results)} graph results.",
                "data": results,
                "cypher": cypher_query
            }
            
        except Exception as e:
            logger.error(f"Graph agent failed: {e}")
            return {"answer": "Error querying knowledge graph.", "error": str(e)}
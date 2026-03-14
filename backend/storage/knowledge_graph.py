"""
Knowledge Graph interaction using Neo4j.
"""
from typing import List, Dict, Any
from neo4j import GraphDatabase
from loguru import logger
from backend.config import settings

class KnowledgeGraph:
    def __init__(self):
        self.driver = None
        self._init_driver()
        
    def _init_driver(self):
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def connect():
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                keep_alive=True  # Important for cloud connections
            )
            self._verify_connection()
            logger.info("Connected to Neo4j")
            
        try:
            connect()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j after retries: {e}")
            
    def _verify_connection(self):
        with self.driver.session() as session:
            session.run("RETURN 1")
            
    def close(self):
        if self.driver:
            self.driver.close()
            
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute explicit Cypher query"""
        from tenacity import retry, stop_after_attempt, wait_exponential

        if not self.driver:
            return []
            
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
        def run_query():
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        try:
            return run_query()
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            return []

    def add_paper(self, paper_data: Dict):
        """
        Add paper and its extracted entities to the graph.
        
        Expected structure:
        {
            "title": "...",
            "authors": ["..."],
            "concepts": ["..."],
            "relationships": [{"source": "...", "target": "...", "relationship": "..."}]
        }
        """
        if not self.driver:
            return

        with self.driver.session() as session:
            # 1. Create Paper Node
            session.run("""
                MERGE (p:Paper {title: $title})
                ON CREATE SET p.year = $year, p.url = $url
            """, {
                "title": paper_data.get("title", "Unknown"), 
                "year": paper_data.get("year"),
                "url": paper_data.get("url")
            })
            
            # 2. Add Authors
            for author in paper_data.get("authors", []):
                session.run("""
                    MERGE (a:Author {name: $name})
                    WITH a
                    MATCH (p:Paper {title: $title})
                    MERGE (a)-[:WROTE]->(p)
                """, {"name": author, "title": paper_data.get("title")})
                
            # 3. Add Concepts
            for concept in paper_data.get("concepts", []):
                session.run("""
                    MERGE (c:Concept {name: $name})
                    WITH c
                    MATCH (p:Paper {title: $title})
                    MERGE (p)-[:HAS_CONCEPT]->(c)
                """, {"name": concept, "title": paper_data.get("title")})
                
            # 4. Add Extracted Relationships
            for rel in paper_data.get("relationships", []):
                # We assume source/target are concepts for simplicity
                session.run("""
                    MERGE (s:Concept {name: $source})
                    MERGE (t:Concept {name: $target})
                    MERGE (s)-[:RELATED {type: $rel}]->(t)
                """, {
                    "source": rel.get("source"),
                    "target": rel.get("target"),
                    "rel": rel.get("relationship", "RELATED_TO")
                })
        
        logger.info(f"Added paper to graph: {paper_data.get('title')}")

    def find_papers_by_concept(self, concept: str) -> List[Dict]:
        query = """
        MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept)
        WHERE toLower(c.name) CONTAINS toLower($concept)
        RETURN p.title as title, p.year as year
        LIMIT 10
        """
        return self.execute_query(query, {"concept": concept})
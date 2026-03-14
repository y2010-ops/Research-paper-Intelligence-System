"""
Entity Extraction using LLM.
Extracts Concepts, Authors, and Relationships from text.
"""
import json
from typing import Dict, List, Any
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.utils.llm_factory import get_llm

class EntityExtractor:
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        
        # Optimized prompt for JSON extraction (works better with smaller models like Llama3)
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert NLP system. Extract entities and relationships from the text below.
        
        Return valid JSON only. Do not add any markdown formatting or explanation.
        
        JSON Structure:
        {{
            "concepts": ["concept1", "concept2"],
            "authors": ["author1", "author2"],
            "relationships": [
                {{"source": "concept1", "relationship": "improves", "target": "concept2"}},
                {{"source": "author1", "relationship": "wrote_paper_on", "target": "concept1"}}
            ]
        }}
        
        Text:
        {text}
        
        JSON Output:
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from text.
        """
        try:
            # Clean JSON response
            response = await self.chain.ainvoke({"text": text[:2000]}) # Limit text length
            
            # Robust parsing: find first '{' and last '}'
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start != -1 and end != -1:
                cleaned_response = response[start:end]
                data = json.loads(cleaned_response)
                
                # Ensure keys exist
                return {
                    "concepts": data.get("concepts", []),
                    "authors": data.get("authors", []),
                    "relationships": data.get("relationships", [])
                }
            else:
                logger.warning(f"No JSON found in response: {response[:100]}...")
                return {"concepts": [], "authors": [], "relationships": []}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response[:100]}... Error: {e}") 
            return {"concepts": [], "authors": [], "relationships": []}
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"concepts": [], "authors": [], "relationships": []}
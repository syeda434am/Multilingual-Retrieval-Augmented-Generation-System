import logging
from typing import List, Dict, Any, Optional
from com.mhire.app.database.db_connection.db_connection import DBConnection
from com.mhire.app.utils.embedding_utility.embedding_create import EmbeddingCreator

logger = logging.getLogger(__name__)

class EmbeddingRetriever:
    def __init__(self):
        self.db_connection = DBConnection()
        self.embedding_creator = EmbeddingCreator()
        self.collection = self.db_connection.collection
        
    async def retrieve_similar_documents(
        self, 
        query_text: str, 
        limit: int = 5,  # Reduced from 10 to 5
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents using vector search for RAG implementation.
        
        Args:
            query_text: The user's query text
            limit: Number of documents to retrieve (reduced to 5)
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            logger.info(f"Retrieving documents for query: {query_text[:100]}...")
            
            # Create query embedding
            query_embedding = await self.embedding_creator.create_query_embedding(query_text)
            
            # Vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 20,  # More candidates for better results
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "file_name": 1,
                        "text": 1,
                        "language_detected": 1,
                        "chunk_index": 1,
                        "total_chunks": 1,
                        "created_at": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": similarity_threshold}
                    }
                },
                {
                    "$sort": {"score": -1}
                }
            ]
            
            results = []
            async for doc in self.collection.aggregate(pipeline):
                result = {
                    "file_name": doc["file_name"],
                    "text": doc["text"],
                    "language_detected": doc.get("language_detected", "unknown"),
                    "chunk_index": doc.get("chunk_index", 0),
                    "total_chunks": doc.get("total_chunks", 1),
                    "similarity_score": doc["score"],
                    "created_at": doc["created_at"]
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    async def retrieve_context_for_rag(
        self, 
        query_text: str, 
        max_context_length: int = 25000  # Increased to accommodate full documents
    ) -> Dict[str, Any]:
        """
        Retrieve and format context for RAG (Retrieval Augmented Generation).
        NO TRUNCATION: Include full documents to preserve answers
        
        Args:
            query_text: The user's query
            max_context_length: Maximum length of combined context (increased)
            
        Returns:
            Dictionary containing formatted context and metadata
        """
        try:
            # Retrieve only 5 relevant documents to keep context manageable
            documents = await self.retrieve_similar_documents(
                query_text=query_text,
                limit=5,  # Reduced from 10 to 5
                similarity_threshold=0.4
            )
            
            logger.info(f"Found {len(documents)} documents for context formatting")
            
            if not documents:
                logger.warning("No documents found for RAG context")
                return {
                    "context": "",
                    "sources": [],
                    "total_documents": 0,
                    "languages_detected": []
                }
            
            # Format context - NO TRUNCATION, include all documents fully
            context_parts = []
            sources = []
            languages = set()
            
            for i, doc in enumerate(documents):
                # Collect metadata for each document
                source_info = {
                    "file_name": doc["file_name"],
                    "chunk_index": doc.get("chunk_index", 0),
                    "similarity_score": round(doc["similarity_score"], 4),
                    "language": doc["language_detected"]
                }
                
                # Add document info with clear formatting
                doc_text = doc["text"].strip()
                chunk_info = f"[Chunk {doc.get('chunk_index', 0)+1}/{doc.get('total_chunks', 1)}]"
                
                # Create well-formatted document entry - FULL CONTENT, NO TRUNCATION
                doc_info = f"=== Document {i+1} from {doc['file_name']} {chunk_info} ===\n{doc_text}\n\n"
                
                # Add full document (no length checking, no truncation)
                context_parts.append(doc_info)
                sources.append(source_info)
                languages.add(doc["language_detected"])
                
                logger.info(f"Added full document {i+1} ({len(doc_text)} chars), total sources: {len(sources)}")
            
            # Combine context
            formatted_context = "".join(context_parts).strip()
            
            logger.info(f"Formatted context: {len(formatted_context)} characters from {len(sources)} sources")
            
            result = {
                "context": formatted_context,
                "sources": sources,
                "total_documents": len(sources),
                "languages_detected": list(languages),
                "context_length": len(formatted_context)
            }
            
            # Debug logging - FIXED syntax error
            if formatted_context:
                logger.info(f"Context preview: {formatted_context[:200]}...")
                source_list = [f"{s['file_name']} chunk {s['chunk_index']}" for s in sources]
                logger.info(f"Sources: {source_list}")
            else:
                logger.warning("Formatted context is empty!")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve context for RAG: {e}")
            return {
                "context": "",
                "sources": [],
                "total_documents": 0,
                "languages_detected": [],
                "error": str(e)
            }
    
    async def test_retrieval(self, query_text: str) -> Dict[str, Any]:
        """
        Test retrieval functionality - for debugging and testing purposes.
        
        Args:
            query_text: Test query
            
        Returns:
            Test results with detailed information
        """
        try:
            logger.info(f"Testing retrieval for query: {query_text}")
            
            # Get RAG context
            rag_result = await self.retrieve_context_for_rag(query_text)
            
            # Get raw documents for comparison
            raw_documents = await self.retrieve_similar_documents(query_text, limit=5)
            
            return {
                "query": query_text,
                "rag_context": rag_result,
                "raw_documents_count": len(raw_documents),
                "raw_documents": raw_documents[:3],  # First 3 for preview
                "test_status": "success"
            }
            
        except Exception as e:
            logger.error(f"Retrieval test failed: {e}")
            return {
                "query": query_text,
                "test_status": "failed",
                "error": str(e)
            }
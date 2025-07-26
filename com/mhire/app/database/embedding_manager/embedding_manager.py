import logging
import time
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from com.mhire.app.database.db_connection.db_connection import DBConnection
from com.mhire.app.utils.embedding_utility.embedding_create import EmbeddingCreator
from com.mhire.app.utils.embedding_utility.embedding_retrieve import EmbeddingRetriever
from com.mhire.app.database.embedding_manager.embedding_manager_schema import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    ChunkedEmbeddingResponse,
    EmbeddingDocument
)

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        self.db_connection = DBConnection()
        self.embedding_creator = EmbeddingCreator()
        self.embedding_retriever = EmbeddingRetriever()
        self.collection = self.db_connection.collection
        
    async def create_vector_index(self):
        """Create vector search index for embeddings"""
        try:
            # Check if index already exists
            indexes = await self.collection.list_indexes().to_list(length=None)
            index_names = [index['name'] for index in indexes]
            
            if 'vector_index' not in index_names:
                index_definition = {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 1536,  # text-embedding-3-small dimensions
                            "similarity": "cosine"
                        },
                        {
                            "type": "filter",
                            "path": "file_name"
                        },
                        {
                            "type": "filter", 
                            "path": "language_detected"
                        },
                        {
                            "type": "filter",
                            "path": "chunk_index"
                        }
                    ]
                }
                
                await self.collection.create_search_index(
                    "vector_index",
                    "vectorSearch", 
                    index_definition
                )
                logger.info("Vector search index created successfully")
            else:
                logger.info("Vector search index already exists")
                
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            # Don't raise exception as the service can work without index initially

    async def process_embedding_request(self, request: EmbeddingRequest) -> ChunkedEmbeddingResponse:
        """Process embedding request with chunking - stores ALL chunks of the document"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing chunked embedding for file: {request.file_name}")
            
            # First, delete any existing chunks for this file
            await self.delete_all_chunks(request.file_name)
            
            # Create embeddings for all chunks
            chunk_embeddings, language = await self.embedding_creator.create_embeddings_for_chunks(request.text)
            
            total_chunks = len(chunk_embeddings)
            successful_chunks = 0
            failed_chunks = 0
            embedding_ids = []
            
            # Store each chunk as a separate document
            for embedding, chunk_text, chunk_index in chunk_embeddings:
                try:
                    document = EmbeddingDocument(
                        file_name=request.file_name,
                        text=chunk_text,
                        embedding=embedding,
                        text_length=len(chunk_text),
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        language_detected=language,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    
                    # Insert into database
                    result = await self.collection.insert_one(document.dict())
                    embedding_ids.append(str(result.inserted_id))
                    successful_chunks += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store chunk {chunk_index}: {e}")
                    failed_chunks += 1
            
            processing_time = round(time.time() - start_time, 2)
            
            success = successful_chunks > 0
            message = f"Processed {successful_chunks}/{total_chunks} chunks successfully"
            
            if failed_chunks > 0:
                message += f" ({failed_chunks} failed)"
            
            logger.info(f"Chunked embedding processing completed for {request.file_name}: {message}")
            
            return ChunkedEmbeddingResponse(
                success=success,
                message=message,
                file_name=request.file_name,
                total_chunks=total_chunks,
                successful_chunks=successful_chunks,
                failed_chunks=failed_chunks,
                embedding_ids=embedding_ids,
                vector_dimensions=len(chunk_embeddings[0][0]) if chunk_embeddings else None,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            error_msg = f"Failed to process chunked embedding: {str(e)}"
            logger.error(error_msg)
            
            return ChunkedEmbeddingResponse(
                success=False,
                message=error_msg,
                file_name=request.file_name,
                total_chunks=0,
                successful_chunks=0,
                failed_chunks=0,
                processing_time=processing_time
            )

    async def update_embedding(self, file_name: str, new_text: str) -> EmbeddingResponse:
        """Update existing embedding for a file"""
        start_time = time.time()
        
        try:
            logger.info(f"Updating embedding for file: {file_name}")
            
            # Create new embedding
            embedding, language = await self.embedding_creator.create_embedding(new_text)
            
            # Update document in database
            update_data = {
                "text": new_text,
                "embedding": embedding,
                "text_length": len(new_text),
                "language_detected": language,
                "updated_at": datetime.utcnow()
            }
            
            result = await self.collection.update_one(
                {"file_name": file_name},
                {"$set": update_data}
            )
            
            processing_time = round(time.time() - start_time, 2)
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated embedding for {file_name}")
                return EmbeddingResponse(
                    success=True,
                    message="Embedding updated successfully",
                    file_name=file_name,
                    vector_dimensions=len(embedding),
                    processing_time=processing_time
                )
            else:
                return EmbeddingResponse(
                    success=False,
                    message=f"No document found with file_name: {file_name}",
                    file_name=file_name,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            error_msg = f"Failed to update embedding: {str(e)}"
            logger.error(error_msg)
            
            return EmbeddingResponse(
                success=False,
                message=error_msg,
                file_name=file_name,
                processing_time=processing_time
            )

    async def retrieve_embeddings(self, query_text: str) -> dict:
        """Retrieve embeddings using embedding utility for RAG testing"""
        try:
            logger.info(f"Retrieving embeddings for query: {query_text[:100]}...")
            
            # Use embedding retriever utility
            result = await self.embedding_retriever.test_retrieval(query_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings: {e}")
            return {
                "query": query_text,
                "test_status": "failed",
                "error": str(e)
            }

    
    async def delete_all_chunks(self, file_name: str) -> int:
        """Delete all chunks for a given file name"""
        try:
            result = await self.collection.delete_many({"file_name": file_name})
            deleted_count = result.deleted_count
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} existing chunks for {file_name}")
            
            return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete chunks for {file_name}: {e}")
            return 0

    async def delete_embedding(self, file_name: str) -> bool:
        """Delete all embeddings/chunks by file name"""
        try:
            result = await self.collection.delete_many({"file_name": file_name})
            
            if result.deleted_count > 0:
                logger.info(f"Successfully deleted {result.deleted_count} chunks for {file_name}")
                return True
            else:
                logger.warning(f"No embeddings found for {file_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return False
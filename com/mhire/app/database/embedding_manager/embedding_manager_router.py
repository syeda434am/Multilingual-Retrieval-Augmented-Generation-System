import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from com.mhire.app.common.network_responses import NetworkResponse, HTTPCode
from com.mhire.app.database.embedding_manager.embedding_manager import EmbeddingManager
from com.mhire.app.database.embedding_manager.embedding_manager_schema import (
    EmbeddingRequest,
    EmbeddingResponse,
    ChunkedEmbeddingResponse,
    EmbeddingSearchRequest
)

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(
    prefix="/api/v1",
    tags=["Embeddings"],
    responses={404: {"description": "Not found"}}
)

# Initialize embedding manager
embedding_manager = EmbeddingManager()

@router.post("/create", response_model=dict)
async def create_embedding(request: EmbeddingRequest):
    """
    Create vector embeddings for ALL chunks of the provided text and store in database.
    This ensures no content is lost from large documents.
    Supports both Bengali and English text, including mixed content.
    """
    start_time = time.time()
    network_response = NetworkResponse()
    
    try:
        logger.info(f"Received embedding request for file: {request.file_name}")
        
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Text content cannot be empty"
            )
        
        if not request.file_name.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="File name cannot be empty"
            )
        
        # Process embedding with chunking
        result = await embedding_manager.process_embedding_request(request)
        
        if result.success:
            return network_response.success_response(
                http_code=HTTPCode.CREATED,
                message=result.message,
                data={
                    "file_name": result.file_name,
                    "total_chunks": result.total_chunks,
                    "successful_chunks": result.successful_chunks,
                    "failed_chunks": result.failed_chunks,
                    "embedding_ids": result.embedding_ids,
                    "vector_dimensions": result.vector_dimensions,
                    "processing_time": f"{result.processing_time}s"
                },
                resource="/api/v1/create",
                start_time=start_time
            )
        else:
            return network_response.json_response(
                http_code=HTTPCode.INTERNAL_SERVER_ERROR,
                error_message=result.message,
                resource="/api/v1/create",
                start_time=start_time
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_embedding: {e}")
        return network_response.json_response(
            http_code=HTTPCode.INTERNAL_SERVER_ERROR,
            error_message=f"Internal server error: {str(e)}",
            resource="/api/v1/create",
            start_time=start_time
        )

@router.put("/update/{file_name}", response_model=dict)
async def update_embedding(file_name: str, request: dict):
    """
    Update existing embedding for a file with new text content.
    """
    start_time = time.time()
    network_response = NetworkResponse()
    
    try:
        logger.info(f"Received update request for file: {file_name}")
        
        # Validate input
        if "text" not in request or not request["text"].strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Text content is required and cannot be empty"
            )
        
        # Update embedding
        result = await embedding_manager.update_embedding(file_name, request["text"])
        
        if result.success:
            return network_response.success_response(
                http_code=HTTPCode.SUCCESS,
                message=result.message,
                data={
                    "file_name": result.file_name,
                    "vector_dimensions": result.vector_dimensions,
                    "processing_time": f"{result.processing_time}s"
                },
                resource=f"/api/v1/update/{file_name}",
                start_time=start_time
            )
        else:
            status_code = HTTPCode.NOT_FOUND if "not found" in result.message.lower() else HTTPCode.INTERNAL_SERVER_ERROR
            return network_response.json_response(
                http_code=status_code,
                error_message=result.message,
                resource=f"/api/v1/update/{file_name}",
                start_time=start_time
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_embedding: {e}")
        return network_response.json_response(
            http_code=HTTPCode.INTERNAL_SERVER_ERROR,
            error_message=f"Internal server error: {str(e)}",
            resource=f"/api/v1/update/{file_name}",
            start_time=start_time
        )

@router.post("/retrieve", response_model=dict)
async def retrieve_embeddings(request: EmbeddingSearchRequest):
    """
    Retrieve similar embeddings for RAG (Retrieval Augmented Generation).
    Supports both Bengali and English query text.
    """
    start_time = time.time()
    network_response = NetworkResponse()
    
    try:
        logger.info(f"Received retrieve request for query: {request.query_text[:50]}...")
        
        # Validate input
        if not request.query_text.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Query text cannot be empty"
            )
        
        # Retrieve embeddings
        test_result = await embedding_manager.retrieve_embeddings(request.query_text)
        
        return network_response.success_response(
            http_code=HTTPCode.SUCCESS,
            message=f"Retrieved {test_result.get('rag_context', {}).get('total_documents', 0)} relevant documents",
            data={
                "query": request.query_text,
                "rag_context": test_result.get("rag_context", {}),
                "raw_documents_count": test_result.get("raw_documents_count", 0),
                "test_status": test_result.get("test_status", "unknown")
            },
            resource="/api/v1/retrieve",
            start_time=start_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in retrieve_embeddings: {e}")
        return network_response.json_response(
            http_code=HTTPCode.INTERNAL_SERVER_ERROR,
            error_message=f"Internal server error: {str(e)}",
            resource="/api/v1/retrieve",
            start_time=start_time
        )

@router.delete("/delete/{file_name}", response_model=dict)
async def delete_embedding(file_name: str):
    """
    Delete all embeddings/chunks by file name.
    """
    start_time = time.time()
    network_response = NetworkResponse()
    
    try:
        logger.info(f"Deleting embedding for file: {file_name}")
        
        deleted = await embedding_manager.delete_embedding(file_name)
        
        if deleted:
            return network_response.success_response(
                http_code=HTTPCode.SUCCESS,
                message=f"All chunks for {file_name} deleted successfully",
                data={"file_name": file_name, "deleted": True},
                resource=f"/api/v1/delete/{file_name}",
                start_time=start_time
            )
        else:
            return network_response.json_response(
                http_code=HTTPCode.NOT_FOUND,
                error_message=f"No embeddings found for file: {file_name}",
                resource=f"/api/v1/delete/{file_name}",
                start_time=start_time
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in delete_embedding: {e}")
        return network_response.json_response(
            http_code=HTTPCode.INTERNAL_SERVER_ERROR,
            error_message=f"Internal server error: {str(e)}",
            resource=f"/api/v1/delete/{file_name}",
            start_time=start_time
        )
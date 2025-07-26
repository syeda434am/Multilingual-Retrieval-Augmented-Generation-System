from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class EmbeddingRequest(BaseModel):
    file_name: str = Field(..., description="Name of the file being processed")
    text: str = Field(..., description="Text content to create embeddings for")

class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    embedding_id: Optional[str] = None
    vector_dimensions: Optional[int] = None
    processing_time: Optional[float] = None

class ChunkedEmbeddingResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    embedding_ids: List[str] = Field(default_factory=list)
    vector_dimensions: Optional[int] = None
    processing_time: Optional[float] = None

class EmbeddingDocument(BaseModel):
    file_name: str
    text: str
    embedding: List[float]
    text_length: int
    chunk_index: int = Field(default=0, description="Index of the chunk within the document")
    total_chunks: int = Field(default=1, description="Total number of chunks for this document")
    language_detected: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class EmbeddingSearchRequest(BaseModel):
    query_text: str = Field(..., description="Text to search for similar embeddings")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")

class EmbeddingSearchResult(BaseModel):
    file_name: str
    text: str
    similarity_score: float
    created_at: datetime
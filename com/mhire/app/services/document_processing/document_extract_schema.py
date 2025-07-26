from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExtractedTextMetadata(BaseModel):
    """Metadata for extracted text"""
    file_size: Optional[int] = Field(None, description="Size of the file in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
    extraction_method: Optional[str] = Field(None, description="Method used for text extraction")
    processing_time: Optional[float] = Field(None, description="Time taken to process the document")
    chunks_processed: Optional[int] = Field(None, description="Number of chunks processed")

class SuccessfulFileResult(BaseModel):
    """Result for a successfully processed file"""
    file_name: str = Field(..., description="Name of the processed file")
    text: str = Field(..., description="Extracted text content")

class FailedFileResult(BaseModel):
    """Result for a failed file"""
    file_name: str = Field(..., description="Name of the failed file")
    error: str = Field(..., description="Error message")

class ExtractExtractionData(BaseModel):
    """Data structure for extraction response"""
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: List[SuccessfulFileResult] = Field(default_factory=list, description="List of successfully processed files")
    failed_files: List[FailedFileResult] = Field(default_factory=list, description="List of failed files")

class DocumentExtractionResponse(BaseModel):
    """Complete response for document extraction request matching NetworkResponse structure"""
    success: bool = Field(..., description="Overall success status")
    message: str = Field(..., description="Status message")
    data: ExtractExtractionData = Field(..., description="Extraction results")
    resource: str = Field(..., description="API endpoint resource path")
    duration: str = Field(..., description="Processing duration")

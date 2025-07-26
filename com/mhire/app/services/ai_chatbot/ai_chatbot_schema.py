from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message/question")
    session_id: str = Field(..., description="Unique session identifier for conversation history")

class ChatResponse(BaseModel):
    success: bool
    message: str
    response: Optional[str] = None
    session_id: str
    language_detected: Optional[str] = None
    sources_used: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None

class SessionHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    language_preference: Optional[str] = None

class RAGContext(BaseModel):
    context: str
    sources: List[Dict[str, Any]]
    total_documents: int
    languages_detected: List[str]

# RAG Evaluation Schemas
class ChatEvaluationRequest(BaseModel):
    message: str = Field(..., description="User's message/question")
    session_id: str = Field(..., description="Unique session identifier")
    expected_answer: str = Field(..., description="Expected correct answer for evaluation")

class GroundednessEvaluation(BaseModel):
    score: float = Field(..., description="Groundedness score (0.0-1.0)")
    analysis: str = Field(..., description="Analysis of how well the answer is supported by context")
    supported: bool = Field(..., description="Whether the answer is supported by context")

class RelevanceEvaluation(BaseModel):
    score: float = Field(..., description="Relevance score (0.0-1.0)")
    analysis: str = Field(..., description="Analysis of document relevance")
    relevant_docs: int = Field(..., description="Number of relevant documents")
    total_docs: int = Field(..., description="Total number of retrieved documents")
    individual_scores: Optional[List[float]] = None

class ChatEvaluationResponse(BaseModel):
    success: bool
    message: str
    actual_answer: Optional[str] = None
    session_id: str
    overall_score: Optional[float] = None
    quality: Optional[str] = None
    groundedness: Optional[GroundednessEvaluation] = None
    relevance: Optional[RelevanceEvaluation] = None
    processing_time: Optional[float] = None
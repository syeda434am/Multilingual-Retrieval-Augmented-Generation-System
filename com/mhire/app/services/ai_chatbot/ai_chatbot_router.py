import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from com.mhire.app.common.network_responses import NetworkResponse, HTTPCode
from com.mhire.app.services.ai_chatbot.ai_chatbot import AIChatbot
from com.mhire.app.services.ai_chatbot.ai_chatbot_schema import (
    ChatRequest,
    ChatResponse,
    SessionHistory,
    ChatEvaluationRequest,
    ChatEvaluationResponse
)

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(
    prefix="/api/v1",
    tags=["AI Chatbot"],
    responses={404: {"description": "Not found"}}
)

# Initialize AI chatbot
ai_chatbot = AIChatbot()

@router.post("/chat", response_model=dict)
async def chat_with_ai(request: ChatRequest):
    """
    Chat with AI using RAG (Retrieval Augmented Generation).
    Supports Bengali, English, and mixed (Banglish) conversations.
    Maintains session history for context-aware responses.
    """
    start_time = time.time()
    network_response = NetworkResponse()
    
    try:
        logger.info(f"Received chat request for session: {request.session_id}")
        
        # Validate input
        if not request.message.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        if not request.session_id.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Session ID cannot be empty"
            )
        
        # Process chat request
        result = await ai_chatbot.process_chat_request(request)
        
        if result.success:
            return network_response.success_response(
                http_code=HTTPCode.SUCCESS,
                message=result.message,
                data={
                    "response": result.response,
                    "session_id": result.session_id,
                    "language_detected": result.language_detected,
                    "sources_used": result.sources_used,
                    "processing_time": f"{result.processing_time}s"
                },
                resource="/api/v1/chat",
                start_time=start_time
            )
        else:
            return network_response.json_response(
                http_code=HTTPCode.INTERNAL_SERVER_ERROR,
                error_message=result.message,
                resource="/api/v1/chat",
                start_time=start_time
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_ai: {e}")
        return network_response.json_response(
            http_code=HTTPCode.INTERNAL_SERVER_ERROR,
            error_message=f"Internal server error: {str(e)}",
            resource="/api/v1/chat",
            start_time=start_time
        )

@router.post("/chat/evaluation", response_model=dict)
async def evaluate_chat_with_ai(request: ChatEvaluationRequest):
    """
    Evaluate RAG system performance with comprehensive metrics.
    
    This endpoint provides:
    - Groundedness: Is the answer supported by retrieved context?
    - Relevance: Does the system fetch the most appropriate documents?
    
    Supports Bengali, English, and mixed (Banglish) conversations.
    """
    start_time = time.time()
    network_response = NetworkResponse()
    
    try:
        logger.info(f"Received chat evaluation request for session: {request.session_id}")
        
        # Validate input
        if not request.message.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        if not request.session_id.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Session ID cannot be empty"
            )
        
        if not request.expected_answer.strip():
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="Expected answer cannot be empty for evaluation"
            )
        
        # Process chat evaluation
        result = await ai_chatbot.process_chat_evaluation(request)
        
        if result.success:
            return network_response.success_response(
                http_code=HTTPCode.SUCCESS,
                message=result.message,
                data={
                    "actual_answer": result.actual_answer,
                    "session_id": result.session_id,
                    "overall_score": result.overall_score,
                    "quality": result.quality,
                    "groundedness": {
                        "score": result.groundedness.score,
                        "analysis": result.groundedness.analysis,
                        "supported": result.groundedness.supported
                    } if result.groundedness else None,
                    "relevance": {
                        "score": result.relevance.score,
                        "analysis": result.relevance.analysis,
                        "relevant_docs": result.relevance.relevant_docs,
                        "total_docs": result.relevance.total_docs
                    } if result.relevance else None,
                    "processing_time": f"{result.processing_time}s"
                },
                resource="/api/v1/chat/evaluation",
                start_time=start_time
            )
        else:
            return network_response.json_response(
                http_code=HTTPCode.INTERNAL_SERVER_ERROR,
                error_message=result.message,
                resource="/api/v1/chat/evaluation",
                start_time=start_time
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in evaluate_chat_with_ai: {e}")
        return network_response.json_response(
            http_code=HTTPCode.INTERNAL_SERVER_ERROR,
            error_message=f"Internal server error: {str(e)}",
            resource="/api/v1/chat/evaluation",
            start_time=start_time
        )
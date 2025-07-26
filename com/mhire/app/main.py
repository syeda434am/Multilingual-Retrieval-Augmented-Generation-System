import logging
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from com.mhire.app.common.network_responses import NetworkResponse, HTTPCode
from com.mhire.app.services.document_processing.document_extract_router import router as extract_router
from com.mhire.app.database.embedding_manager.embedding_manager_router import router as embedding_router
from com.mhire.app.services.ai_chatbot.ai_chatbot_router import router as chatbot_router
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI-Extractor",
    description="AI-powered document extraction and description generation service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(extract_router)
app.include_router(embedding_router)
app.include_router(chatbot_router)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(http_request: Request, exc: HTTPException):
    """Handle HTTP exceptions and return network response format"""
    start_time = time.time()
    network_response = NetworkResponse()
    return network_response.json_response(
        http_code=exc.status_code,
        error_message=str(exc.detail),
        resource=http_request.url.path,
        start_time=start_time
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors and return network response format"""
    start_time = time.time()
    network_response = NetworkResponse()
    return network_response.json_response(
        http_code=HTTPCode.BAD_REQUEST,
        error_message=f"Validation error: {str(exc)}",
        resource=request.url.path,
        start_time=start_time
    )

@app.get("/")
async def root():
    start_time = time.time()
    network_response = NetworkResponse()
    return network_response.success_response(
        http_code=HTTPCode.SUCCESS,
        message="API is running",
        data={"status": "active"},
        resource="/",
        start_time=start_time
)
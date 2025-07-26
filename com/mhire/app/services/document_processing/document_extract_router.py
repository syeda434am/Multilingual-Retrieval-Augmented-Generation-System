import time
import tempfile
import os
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from com.mhire.app.common.network_responses import NetworkResponse, HTTPCode
from com.mhire.app.services.document_processing.document_extract import DocumentProcessor
from com.mhire.app.services.document_processing.document_extract_schema import DocumentExtractionResponse

router = APIRouter(prefix="/api/v1", tags=["document-extraction"])
network_response = NetworkResponse()

@router.post("/extract", response_model=DocumentExtractionResponse)
async def extract_text(http_request: Request, files: List[UploadFile] = File(...)):
    """
    Extract text from one or more documents using Document AI.
    Supports various file formats and automatically converts them to PDF.
    """
    start_time = time.time()
    
    try:
        # Validate files
        if not files:
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="No files provided"
            )

        # Initialize processor
        processor = DocumentProcessor()
        
        # Use temporary directory instead of creating uploads folder
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_files = []
            
            # Save uploaded files to temporary directory
            for file in files:
                if not file.filename:
                    raise HTTPException(
                        status_code=HTTPCode.BAD_REQUEST,
                        detail="File must have a filename"
                    )
                
                file_path = os.path.join(temp_dir, file.filename)
                try:
                    # Save file
                    with open(file_path, "wb") as buffer:
                        content = await file.read()
                        buffer.write(content)
                    processed_files.append(file_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=HTTPCode.INTERNAL_SERVER_ERROR,
                        detail=f"Error saving file {file.filename}: {str(e)}"
                    )

            # Process all files
            result = processor.process_multiple_files(processed_files)

            # Format response data
            response_data = {
                'total_files': len(processed_files),
                'failed_files': [],
                'successful_files': []
            }

            # Organize results into successful and failed files
            for file_result in result['individual_results']:
                if file_result['success']:
                    response_data['successful_files'].append({
                        'file_name': file_result['filename'],
                        'text': file_result['extracted_text']
                    })
                else:
                    response_data['failed_files'].append({
                        'file_name': file_result['filename'],
                        'error': file_result.get('error', 'Unknown error')
                    })

            if result['success']:
                return network_response.success_response(
                    http_code=HTTPCode.SUCCESS,
                    message="Text extraction completed successfully",
                    data=response_data,
                    resource=http_request.url.path,
                    start_time=start_time
                )
            else:
                # If all files failed, return error
                if result['successful_files'] == 0:
                    raise HTTPException(
                        status_code=HTTPCode.UNPROCESSABLE_ENTITY,
                        detail="All files failed to process"
                    )
                else:
                    # Some files succeeded, some failed - return partial success
                    return network_response.success_response(
                        http_code=HTTPCode.SUCCESS,
                        message="Text extraction completed with some failures",
                        data=response_data,
                        resource=http_request.url.path,
                        start_time=start_time
                    )

    except HTTPException:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTPCode.INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}"
        )
import os
import mimetypes
from typing import Optional, Dict, Any
from fastapi import HTTPException
from com.mhire.app.common.network_responses import HTTPCode
from com.mhire.app.utils.gcp_utility.gcp_util import GCPUtil

class TextExtractor:
    """Factory class for text extraction operations using Google Document AI via GCP Util"""
    
    def __init__(self):
        self.gcp_util = GCPUtil()
    
    def process_document(self, file_path: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process document and extract text using Google Document AI via GCP Util
        Returns a dictionary with extraction results
        """
        
        try:
            print(f"    Processing document: {os.path.basename(file_path)}")
            
            # Get MIME type if not provided
            if not mime_type:
                mime_type = self.get_mime_type(file_path)
            
            print(f"    MIME type: {mime_type}")
            
            # Handle text files directly
            if mime_type == 'text/plain' or file_path.lower().endswith('.txt'):
                return self._process_text_file_directly(file_path)
            
            print(f"    Reading file: {os.path.basename(file_path)}")
            # Read file content
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            print(f"    File size: {len(file_content)} bytes")
            print(f"    Sending request to Document AI...")
            
            # Use GCP Util to process the document
            result = self.gcp_util.process_document(file_content, mime_type)
            
            if result['success']:
                print(f"    Document AI processing completed")
                # Add file-specific metadata
                result['metadata']['file_path'] = file_path
                result['metadata']['file_size'] = len(file_content)
            
            return result
            
        except Exception as e:
            print(f"    Error in process_document: {str(e)}")
            return {
                'success': False,
                'text': '',
                'error': f"Error processing document: {str(e)}",
                'metadata': {'file_path': file_path}
            }
    
    def _process_text_file_directly(self, file_path: str) -> Dict[str, Any]:
        """Process text files directly without Document AI"""
        try:
            print(f"    Reading text file directly: {os.path.basename(file_path)}")
            
            # Read text file with multiple encoding attempts
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
            text_content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text_content = f.read()
                    print(f"    Successfully read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                raise Exception("Could not read text file with any supported encoding")
            
            if not text_content.strip():
                raise Exception("Text file is empty")
            
            file_size = os.path.getsize(file_path)
            
            return {
                'success': True,
                'text': text_content,
                'error': None,
                'metadata': {
                    'file_path': file_path,
                    'file_size': file_size,
                    'mime_type': 'text/plain',
                    'extraction_method': 'direct_read',
                    'text_length': len(text_content)
                }
            }
            
        except Exception as e:
            print(f"    Error reading text file: {str(e)}")
            return {
                'success': False,
                'text': '',
                'error': f"Error reading text file: {str(e)}",
                'metadata': {'file_path': file_path}
            }
    
    def get_mime_type(self, file_path: str) -> str:
        """Get MIME type for file"""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            extension = os.path.splitext(file_path)[1].lower()
            mime_map = {
                '.pdf': 'application/pdf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff',
                '.bmp': 'image/bmp',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.txt': 'text/plain',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.ppt': 'application/vnd.ms-powerpoint',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel'
            }
            mime_type = mime_map.get(extension, 'application/pdf')
        return mime_type
    
    def validate_document_ai_setup(self) -> Dict[str, Any]:
        """Validate Document AI configuration and connectivity via GCP Util"""
        return self.gcp_util.validate_setup()
    
    def get_supported_formats(self) -> Dict[str, list]:
        """Return list of supported file formats"""
        return {
            'images': ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'],
            'documents': ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls'],
            'text': ['.txt', '.rtf', '.html', '.htm', '.xml', '.json', '.yaml', '.yml'],
            'code': ['.py', '.js', '.css', '.java', '.cpp', '.c', '.sql'],
            'other': ['.csv']
        }
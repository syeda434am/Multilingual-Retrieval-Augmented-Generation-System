import os
from typing import Optional, Dict, Any
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from com.mhire.app.config.config import Config

class GCPUtil:
    
    def __init__(self):
        self.config = Config()
        self.client = None
        self._setup_credentials()
        self._initialize_client()
    
    def _setup_credentials(self):
        """Setup Google Cloud credentials with proper path handling"""
        if self.config.google_application_credential:
            cred_path = self.config.google_application_credential
            
            # Check if we're running in Docker (Unix-style paths) or Windows
            if os.name == 'posix':  # Unix/Linux/Docker
                # Keep Unix-style paths as-is
                if not os.path.isabs(cred_path):
                    cred_path = os.path.join(os.getcwd(), cred_path)
            else:  # Windows
                # Convert Unix path to Windows relative path
                if cred_path.startswith('/etc/'):
                    cred_path = cred_path.replace('/etc/', 'etc\\')
                elif cred_path.startswith('\\etc\\'):
                    cred_path = cred_path.replace('\\etc\\', 'etc\\')
                
                # Make it absolute path from current working directory
                if not os.path.isabs(cred_path):
                    cred_path = os.path.join(os.getcwd(), cred_path)
            
            # Set the environment variable for Google Cloud
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
            print(f"Google Cloud credentials set to: {cred_path}")
            
            # Verify the file exists
            if not os.path.exists(cred_path):
                print(f"WARNING: Credentials file not found at: {cred_path}")
            else:
                print(f"Credentials file found at: {cred_path}")

    def _initialize_client(self) -> None:
        """Initialize Document AI client"""
        try:
            self.client = documentai.DocumentProcessorServiceClient(
                client_options=ClientOptions(
                    api_endpoint=f"{self.config.location}-documentai.googleapis.com"
                )
            )
            print("Document AI client initialized successfully")
        except Exception as e:
            print(f"Error initializing Document AI client: {e}")
            self.client = None

    def process_document(self, content: bytes, mime_type: str) -> Dict[str, Any]:
        """Process document using Document AI"""
        if not self.client:
            print("    ERROR: Document AI client not initialized")
            return {
                'success': False,
                'text': '',
                'error': 'Document AI client not initialized',
                'metadata': {}
            }

        try:
            print(f"    Building processor path with project_id={self.config.project_id}, location={self.config.location}, processor_id={self.config.processor_id}, processor_version={self.config.processor_version}")
            
            # Get processor path
            name = self.client.processor_version_path(
                self.config.project_id,
                self.config.location,
                self.config.processor_id,
                self.config.processor_version
            )
            print(f"    Processor path: {name}")

            # Configure process options for layout analysis
            process_options = documentai.ProcessOptions(
                layout_config=documentai.ProcessOptions.LayoutConfig(
                    chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                        chunk_size=1000,
                        include_ancestor_headings=True,
                    )
                )
            )

            # Process document
            request = documentai.ProcessRequest(
                name=name,
                raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
                process_options=process_options,
            )

            print(f"    Sending request to Document AI...")
            result = self.client.process_document(request=request)
            document = result.document
            print(f"    Document AI response received")

            # Extract text using multiple methods
            extracted_text = ''
            extraction_method = 'none'

            # Method 1: Direct text
            if document.text and document.text.strip():
                extracted_text = document.text
                extraction_method = 'direct_text'
                print(f"    Extracted text using direct_text method: {len(extracted_text)} characters")

            # Method 2: From chunks
            elif hasattr(document, 'chunked_document') and document.chunked_document:
                if document.chunked_document.chunks:
                    chunk_text = []
                    for chunk in document.chunked_document.chunks:
                        if hasattr(chunk, 'content'):
                            chunk_text.append(chunk.content)
                    if chunk_text:
                        extracted_text = '\n'.join(chunk_text)
                        extraction_method = 'chunked_text'
                        print(f"    Extracted text using chunked_text method: {len(extracted_text)} characters")

            # Method 3: From layout blocks
            elif hasattr(document, 'document_layout') and document.document_layout:
                if document.document_layout.blocks:
                    block_text = []
                    for block in document.document_layout.blocks:
                        if hasattr(block, 'text_block') and block.text_block:
                            if hasattr(block.text_block, 'text'):
                                block_text.append(block.text_block.text)
                    if block_text:
                        extracted_text = '\n'.join(block_text)
                        extraction_method = 'layout_blocks'
                        print(f"    Extracted text using layout_blocks method: {len(extracted_text)} characters")

            # Method 4: From pages and paragraphs
            if not extracted_text and hasattr(document, 'pages') and document.pages:
                page_text = []
                for page in document.pages:
                    if hasattr(page, 'paragraphs') and page.paragraphs:
                        for paragraph in page.paragraphs:
                            if hasattr(paragraph, 'layout') and paragraph.layout:
                                if hasattr(paragraph.layout, 'text_anchor') and paragraph.layout.text_anchor:
                                    if hasattr(paragraph.layout.text_anchor, 'text_segments'):
                                        for segment in paragraph.layout.text_anchor.text_segments:
                                            if hasattr(segment, 'start_index') and hasattr(segment, 'end_index'):
                                                start = segment.start_index
                                                end = segment.end_index
                                                if document.text:
                                                    page_text.append(document.text[start:end])
                if page_text:
                    extracted_text = '\n'.join(page_text)
                    extraction_method = 'page_paragraphs'
                    print(f"    Extracted text using page_paragraphs method: {len(extracted_text)} characters")

            if not extracted_text:
                print(f"    WARNING: No text could be extracted from document")
                print(f"    Document has text: {bool(document.text)}")
                print(f"    Document text length: {len(document.text) if document.text else 0}")
                print(f"    Document has pages: {bool(hasattr(document, 'pages') and document.pages)}")
                if hasattr(document, 'pages') and document.pages:
                    print(f"    Number of pages: {len(document.pages)}")

            # Prepare metadata
            metadata = {
                'mime_type': mime_type,
                'extraction_method': extraction_method,
                'text_length': len(extracted_text) if extracted_text else 0
            }

            # Add document-specific metadata
            if hasattr(document, 'pages') and document.pages:
                metadata['page_count'] = len(document.pages)

            success = bool(extracted_text and extracted_text.strip())
            
            return {
                'success': success,
                'text': extracted_text or "",
                'error': None if success else "No text could be extracted from the document",
                'metadata': metadata
            }

        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            print(f"    ERROR: {error_msg}")
            return {
                'success': False,
                'text': '',
                'error': error_msg,
                'metadata': {'mime_type': mime_type}
            }

    def validate_setup(self) -> Dict[str, Any]:
        """Validate GCP Document AI setup"""
        validation = {
            'valid': True,
            'issues': [],
            'config_status': {}
        }

        # Check configuration
        required_configs = [
            ('google_application_credential', self.config.google_application_credential),
            ('project_id', self.config.project_id),
            ('processor_id', self.config.processor_id),
            ('location', self.config.location),
            ('processor_version', self.config.processor_version)
        ]

        for config_name, config_value in required_configs:
            validation['config_status'][config_name] = bool(config_value)
            if not config_value:
                validation['valid'] = False
                validation['issues'].append(f"Missing {config_name}")

        # Check client initialization
        if not self.client:
            validation['valid'] = False
            validation['issues'].append("Document AI client not initialized")

        return validation

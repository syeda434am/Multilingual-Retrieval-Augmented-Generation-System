import os
import tempfile
from typing import List, Dict, Any
from fastapi import HTTPException
from com.mhire.app.config.config import Config
from com.mhire.app.common.network_responses import HTTPCode
from com.mhire.app.utils.extraction_utility.conversion_util import DocumentConverter
from com.mhire.app.utils.extraction_utility.extraction_util import TextExtractor
from com.mhire.app.utils.extraction_utility.divide_util import DocumentDivider

class DocumentProcessor:
    """Main processor that coordinates conversion → extract workflow"""
    
    def __init__(self):
        self.config = Config()
        self.converter = DocumentConverter()
        self.extractor = TextExtractor()
        self.divider = DocumentDivider(page_limit=25)  # Set page limit for Document AI
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file through the complete workflow
        Returns detailed processing results
        """
        
        if not os.path.exists(file_path):
            return self._create_error_result(file_path, "File does not exist")
        
        # Initialize result structure
        result = {
            'success': False,
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'extracted_text': '',
            'text_length': 0,
            'processing_steps': [],
            'metadata': {},
            'error': None
        }
        
        try:
            # Step 1: Convert to PDF if needed
            pdf_file_path = self.converter.convert_to_pdf(file_path)
            result['processing_steps'].append({
                'step': 'conversion',
                'success': True,
                'output': pdf_file_path,
                'details': f"Converted to: {os.path.basename(pdf_file_path)}"
            })
            
            # Step 2: Check if PDF needs to be divided into chunks
            file_info = self.divider.get_file_info(pdf_file_path)
            
            # Process based on page count
            if (file_info and 
                'pages' in file_info and 
                isinstance(file_info['pages'], int) and 
                file_info['pages'] > self.divider.page_limit):
                
                # Process large PDF with chunking
                result = self._process_large_pdf(pdf_file_path, result)
            else:
                # Process normally for small PDFs or non-PDFs
                extraction_result = self.extractor.process_document(pdf_file_path)
                result['processing_steps'].append({
                    'step': 'extraction',
                    'success': extraction_result['success'],
                    'output': extraction_result,
                    'details': f"Extraction method: {extraction_result.get('metadata', {}).get('extraction_method', 'unknown')}"
                })
                
                if extraction_result['success']:
                    result['extracted_text'] = extraction_result['text']
                    result['text_length'] = len(extraction_result['text'])
                    result['metadata'] = extraction_result.get('metadata', {})
                    result['success'] = True
                else:
                    result['error'] = extraction_result.get('error', 'Unknown extraction error')
                    result['success'] = False
            
            # Clean up temporary PDF if it was created
            if pdf_file_path != file_path and os.path.exists(pdf_file_path):
                try:
                    os.remove(pdf_file_path)
                except:
                    pass
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
            result['error'] = error_msg
            result['success'] = False
            return result
    
    def _process_large_pdf(self, pdf_file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a large PDF by dividing it into chunks, processing each chunk,
        and then combining the results
        """
        try:
            # Divide the PDF into chunks
            chunk_paths = self.divider.divide_pdf_into_chunks(
                pdf_file_path, 
                self.divider.get_file_info(pdf_file_path)['pages']
            )
            
            result['processing_steps'].append({
                'step': 'division',
                'success': True,
                'output': f"{len(chunk_paths)} chunks created",
                'details': f"PDF divided into {len(chunk_paths)} chunks of max {self.divider.page_limit} pages each"
            })
            
            # Process each chunk
            combined_text = ""
            chunk_results = []
            successful_chunks = 0
            failed_chunks = 0
            
            for i, chunk_path in enumerate(chunk_paths):
                print(f"Processing chunk {i+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")
                
                # Extract text from this chunk
                chunk_extraction_result = self.extractor.process_document(chunk_path)
                
                # Store chunk result
                chunk_result = {
                    'chunk_index': i+1,
                    'chunk_path': chunk_path,
                    'success': chunk_extraction_result['success'],
                    'text_length': len(chunk_extraction_result.get('text', '')),
                    'error': chunk_extraction_result.get('error')
                }
                chunk_results.append(chunk_result)
                
                # Add text from successful chunks
                if chunk_extraction_result['success']:
                    successful_chunks += 1
                    if combined_text and chunk_extraction_result.get('text'):
                        combined_text += "\n\n"
                    combined_text += chunk_extraction_result.get('text', '')
                else:
                    failed_chunks += 1
                
                # Clean up temporary chunk file
                try:
                    os.remove(chunk_path)
                except Exception as e:
                    print(f"Warning: Could not remove temporary chunk file {chunk_path}: {str(e)}")
            
            # Update result with combined information
            result['processing_steps'].append({
                'step': 'chunked_extraction',
                'success': successful_chunks > 0,
                'output': {
                    'total_chunks': len(chunk_paths),
                    'successful_chunks': successful_chunks,
                    'failed_chunks': failed_chunks
                },
                'details': f"Processed {successful_chunks}/{len(chunk_paths)} chunks successfully"
            })
            
            # Set overall result based on chunk processing
            if successful_chunks > 0:
                result['success'] = True
                result['extracted_text'] = combined_text
                result['text_length'] = len(combined_text)
                result['metadata']['chunked_processing'] = True
                result['metadata']['total_chunks'] = len(chunk_paths)
                result['metadata']['successful_chunks'] = successful_chunks
                result['metadata']['failed_chunks'] = failed_chunks
            else:
                result['success'] = False
                result['error'] = "All PDF chunks failed to process"
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing large PDF {os.path.basename(pdf_file_path)}: {str(e)}"
            result['error'] = error_msg
            result['success'] = False
            return result
    
    def process_multiple_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple files and return comprehensive results
        """
        total_files = len(file_paths)
        
        results = {
            'success': True,
            'total_files': total_files,
            'successful_files': 0,
            'failed_files': 0,
            'individual_results': [],
            'combined_text': ''
        }
        
        # Process each file individually
        for i, file_path in enumerate(file_paths, 1):
            file_result = self.process_single_file(file_path)
            results['individual_results'].append(file_result)
            
            if file_result['success']:
                results['successful_files'] += 1
                if file_result['extracted_text']:
                    if results['combined_text']:
                        results['combined_text'] += f"\n\n=== FILE {i}: {file_result['filename']} ===\n\n"
                    else:
                        results['combined_text'] = f"=== FILE {i}: {file_result['filename']} ===\n\n"
                    results['combined_text'] += file_result['extracted_text']
            else:
                results['failed_files'] += 1
        
        # Overall success if at least one file was processed successfully
        if results['failed_files'] > 0:
            results['success'] = results['successful_files'] > 0
        
        return results
    
    def _create_error_result(self, file_path: str, error_message: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            'success': False,
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'extracted_text': '',
            'text_length': 0,
            'processing_steps': [],
            'metadata': {},
            'error': error_message
        }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get overall processing system status"""
        return {
            'converter_available': True,
            'extractor_status': self.extractor.validate_document_ai_setup()
        }
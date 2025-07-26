import os
import tempfile
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfWriter, PdfReader
from com.mhire.app.config.config import Config

class DocumentDivider:
    """Factory class for document division operations"""
    
    def __init__(self, page_limit: int = 25):
        self.config = Config()
        self.page_limit = page_limit
    
    def check_and_divide_file(self, file_path: str) -> List[str]:
        """
        Check if the file needs to be divided based on page count
        Returns list with the original file path or paths to divided chunks
        """
        
        print(f"File processing: {os.path.basename(file_path)}")
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Check if it's a PDF and needs division
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            file_info = self.get_file_info(file_path)
            if file_info and 'pages' in file_info and isinstance(file_info['pages'], int):
                page_count = file_info['pages']
                print(f"PDF has {page_count} pages")
                
                if page_count > self.page_limit:
                    print(f"PDF exceeds page limit of {self.page_limit}, dividing into chunks")
                    return self.divide_pdf_into_chunks(file_path, page_count)
        
        # No division needed - return original file
        print("No file division needed - processing original file")
        return [file_path]
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about a file"""
        
        if not os.path.exists(file_path):
            return None
        
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        info = {
            'path': file_path,
            'size_bytes': file_size,
            'size_mb': file_size / (1024*1024),
            'extension': file_extension
        }
        
        # Add PDF-specific info
        if file_extension == '.pdf':
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    info['pages'] = len(pdf_reader.pages)
            except Exception as e:
                print(f"Error reading PDF: {str(e)}")
                info['pages'] = 'Unknown'
        
        return info
        
    def divide_pdf_into_chunks(self, file_path: str, page_count: int) -> List[str]:
        """Divide a PDF into chunks based on page limit"""
        
        chunk_files = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Calculate number of chunks needed
                num_chunks = (page_count + self.page_limit - 1) // self.page_limit
                print(f"Dividing PDF into {num_chunks} chunks")
                
                for chunk_idx in range(num_chunks):
                    # Create a temporary file for this chunk
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{chunk_idx+1}.pdf")
                    chunk_path = temp_file.name
                    temp_file.close()
                    
                    # Create PDF writer for this chunk
                    pdf_writer = PdfWriter()
                    
                    # Calculate page range for this chunk
                    start_page = chunk_idx * self.page_limit
                    end_page = min((chunk_idx + 1) * self.page_limit, page_count)
                    
                    # Add pages to this chunk
                    for page_idx in range(start_page, end_page):
                        pdf_writer.add_page(pdf_reader.pages[page_idx])
                    
                    # Write chunk to file
                    with open(chunk_path, 'wb') as chunk_file:
                        pdf_writer.write(chunk_file)
                    
                    print(f"Created chunk {chunk_idx+1}/{num_chunks}: {os.path.basename(chunk_path)} with pages {start_page+1}-{end_page}")
                    chunk_files.append(chunk_path)
                
                return chunk_files
                
        except Exception as e:
            print(f"Error dividing PDF: {str(e)}")
            # If division fails, return the original file
            return [file_path]
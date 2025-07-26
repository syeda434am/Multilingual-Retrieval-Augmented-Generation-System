import logging
import openai
import asyncio
import re
from typing import List, Tuple, Optional
from langdetect import detect
from com.mhire.app.config.config import Config

logger = logging.getLogger(__name__)

class EmbeddingCreator:
    def __init__(self):
        self.config = Config()
        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key
        )
        self.model = "text-embedding-3-small"  # Efficient and good for multilingual content
        self.max_tokens = 8000  # Safe limit for the model
        
    def detect_language(self, text: str) -> Optional[str]:
        """Detect if text is Bengali, English, or mixed"""
        try:
            # Count Bengali characters
            bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            total_chars = bengali_chars + english_chars
            
            if total_chars == 0:
                return "unknown"
            
            bengali_ratio = bengali_chars / total_chars
            
            if bengali_ratio > 0.6:
                return "bengali"
            elif bengali_ratio < 0.2:
                return "english"
            else:
                return "mixed"
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better embedding quality - LESS AGGRESSIVE"""
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        
        # Keep more characters to preserve content structure
        # Only remove truly problematic characters, keep punctuation and numbers
        text = re.sub(r'[^\u0980-\u09FF\w\s.,;:!?()\-\[\]০-৯0-9]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, max_length: int = 5500) -> List[str]:
        """Split text into optimal chunks of 5000-6000 characters"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Calculate chunk end position
            chunk_end = min(current_pos + max_length, len(text))
            
            # If this is not the last chunk, try to find a good break point
            if chunk_end < len(text):
                # Look for sentence endings within the last 500 characters
                search_start = max(chunk_end - 500, current_pos)
                chunk_text = text[search_start:chunk_end]
                
                # Find the last sentence ending
                sentence_endings = []
                for pattern in [r'।\s+', r'\.\s+', r'!\s+', r'\?\s+']:
                    for match in re.finditer(pattern, chunk_text):
                        sentence_endings.append(search_start + match.end())
                
                if sentence_endings:
                    # Use the last sentence ending as break point
                    chunk_end = max(sentence_endings)
                else:
                    # If no sentence ending found, look for word boundaries
                    search_text = text[chunk_end-100:chunk_end]
                    word_boundaries = [m.start() for m in re.finditer(r'\s+', search_text)]
                    if word_boundaries:
                        chunk_end = chunk_end - 100 + word_boundaries[-1]
            
            # Extract the chunk
            chunk = text[current_pos:chunk_end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            current_pos = chunk_end
        
        return chunks
    
    async def create_embedding(self, text: str) -> Tuple[List[float], str]:
        """Create embedding for the given text - DEPRECATED: Use create_embeddings_for_chunks instead"""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Detect language
            language = self.detect_language(processed_text)
            
            if not processed_text.strip():
                raise ValueError("Text is empty after preprocessing")
            
            # For backward compatibility, use first chunk only
            chunks = self.chunk_text(processed_text)
            main_text = chunks[0]  # Use first chunk for embedding
            
            logger.info(f"Creating embedding for text of length {len(main_text)}, language: {language}")
            
            response = await self.openai_client.embeddings.create(
                input=main_text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            logger.info(f"Successfully created embedding with {len(embedding)} dimensions")
            
            return embedding, language
            
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise

    async def create_embeddings_for_chunks(self, text: str) -> Tuple[List[Tuple[List[float], str, int]], str]:
        """
        Create embeddings for ALL chunks of the given text
        Returns: (list of (embedding, chunk_text, chunk_index), language)
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Detect language
            language = self.detect_language(processed_text)
            
            if not processed_text.strip():
                raise ValueError("Text is empty after preprocessing")
            
            # Split text into chunks
            chunks = self.chunk_text(processed_text)
            
            logger.info(f"Creating embeddings for {len(chunks)} chunks, total length: {len(processed_text)}, language: {language}")
            
            # Log chunk sizes for debugging
            for i, chunk in enumerate(chunks):
                logger.info(f"Chunk {i+1}: {len(chunk)} characters")
            
            # Create embeddings for all chunks
            chunk_embeddings = []
            
            # Process chunks in batches to avoid rate limits
            batch_size = 10  # Adjust based on API limits
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Create embeddings for this batch
                response = await self.openai_client.embeddings.create(
                    input=batch_chunks,
                    model=self.model
                )
                
                # Store embeddings with chunk info
                for j, embedding_data in enumerate(response.data):
                    chunk_index = i + j
                    embedding = embedding_data.embedding
                    chunk_text = batch_chunks[j]
                    chunk_embeddings.append((embedding, chunk_text, chunk_index))
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully created {len(chunk_embeddings)} embeddings for all chunks")
            
            return chunk_embeddings, language
            
        except Exception as e:
            logger.error(f"Failed to create embeddings for chunks: {e}")
            raise
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[Tuple[List[float], str]]:
        """Create embeddings for multiple texts in batch"""
        try:
            # Process texts
            processed_data = []
            for text in texts:
                processed_text = self.preprocess_text(text)
                language = self.detect_language(processed_text)
                chunks = self.chunk_text(processed_text)
                processed_data.append((chunks[0], language))
            
            # Extract just the text for embedding
            texts_for_embedding = [data[0] for data in processed_data]
            
            response = await self.openai_client.embeddings.create(
                input=texts_for_embedding,
                model=self.model
            )
            
            # Combine embeddings with language info
            results = []
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding
                language = processed_data[i][1]
                results.append((embedding, language))
            
            logger.info(f"Successfully created {len(results)} embeddings")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to create batch embeddings: {e}")
            raise
    
    async def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for search query"""
        try:
            processed_query = self.preprocess_text(query)
            
            if not processed_query.strip():
                raise ValueError("Query is empty after preprocessing")
            
            response = await self.openai_client.embeddings.create(
                input=processed_query,
                model=self.model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}")
            raise
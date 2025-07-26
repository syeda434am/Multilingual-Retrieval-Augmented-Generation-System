import logging
import re
import openai
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from com.mhire.app.config.config import Config

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.config = Config()
        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key
        )
        # Use the model from your config (GPT-4.1)
        self.model = self.config.openai_model or "gpt-3.5-turbo"  # Fallback only if not set
        logger.info(f"RAG Evaluator initialized with model: {self.model}")
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for evaluation"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase for comparison
        text = text.lower().strip()
        return text
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF-IDF"""
        try:
            # Preprocess texts
            text1 = self.preprocess_text(text1)
            text2 = self.preprocess_text(text2)
            
            if not text1 or not text2:
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words=None)  # Keep all words for Bengali support
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def check_keyword_overlap(self, query: str, document: str) -> float:
        """Check keyword overlap between query and document for Bengali text"""
        try:
            # Extract keywords from query and document
            query_words = set(re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+', query.lower()))
            doc_words = set(re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+', document.lower()))
            
            if not query_words:
                return 0.0
            
            # Calculate overlap ratio
            overlap = len(query_words.intersection(doc_words))
            overlap_ratio = overlap / len(query_words)
            
            return overlap_ratio
            
        except Exception as e:
            logger.error(f"Error checking keyword overlap: {e}")
            return 0.0
    
    async def evaluate_groundedness(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate if the answer is grounded in the provided context
        Returns groundedness score and analysis
        """
        try:
            if not context.strip():
                return {
                    "score": 0.0,
                    "analysis": "No context provided",
                    "supported": False
                }
            
            # Use your GPT-4.1 model to evaluate groundedness
            prompt = f"""আপনি একটি RAG সিস্টেম মূল্যায়নকারী। নিচের উত্তরটি প্রদত্ত প্রসঙ্গ দ্বারা সমর্থিত কিনা তা মূল্যায়ন করুন।

প্রসঙ্গ:
{context}

উত্তর:
{answer}

নির্দেশনা:
1. উত্তরটি প্রসঙ্গে উল্লিখিত তথ্য দ্বারা সমর্থিত কিনা বিশ্লেষণ করুন
2. 0.0 থেকে 1.0 স্কেল এ একটি স্কোর দিন (1.0 = সম্পূর্ণ সমর্থিত, 0.0 = সমর্থিত নয়)
3. সংক্ষিপ্ত বিশ্লেষণ প্রদান করুন

উত্তর ফরম্যাট:
স্কোর: [0.0-1.0]
বিশ্লেষণ: [সংক্ষিপ্ত ব্যাখ্যা]"""

            response = await self.openai_client.chat.completions.create(
                model=self.model,  # Use your GPT-4.1 from config
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse the response
            score_match = re.search(r'স্কোর:\s*([0-9.]+)', result_text)
            analysis_match = re.search(r'বিশ্লেষণ:\s*(.+)', result_text, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.0
            analysis = analysis_match.group(1).strip() if analysis_match else result_text
            
            return {
                "score": min(max(score, 0.0), 1.0),  # Ensure score is between 0 and 1
                "analysis": analysis,
                "supported": score >= 0.7
            }
            
        except Exception as e:
            logger.error(f"Error evaluating groundedness: {e}")
            return {
                "score": 0.0,
                "analysis": f"Error in evaluation: {str(e)}",
                "supported": False
            }
    
    def evaluate_relevance(self, query: str, retrieved_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the relevance of retrieved documents to the query
        IMPROVED: Better handling for Bengali text and vector search scores
        """
        try:
            if not retrieved_documents:
                return {
                    "score": 0.0,
                    "analysis": "No documents retrieved",
                    "relevant_docs": 0,
                    "total_docs": 0
                }
            
            query_processed = self.preprocess_text(query)
            relevance_scores = []
            relevant_count = 0
            
            for doc in retrieved_documents:
                doc_text = self.preprocess_text(doc.get('text', ''))
                
                # Calculate cosine similarity
                similarity = self.calculate_cosine_similarity(query_processed, doc_text)
                relevance_scores.append(similarity)
                
                # For Bengali text, use lower threshold and also check keyword overlap
                keyword_overlap = self.check_keyword_overlap(query, doc_text)
                
                # Consider relevant if similarity > 0.1 OR keyword overlap > 0.2 OR similarity_score from vector search > 0.5
                vector_score = doc.get('similarity_score', 0.0)
                
                if similarity > 0.1 or keyword_overlap > 0.2 or vector_score > 0.5:
                    relevant_count += 1
            
            # Calculate overall relevance score
            avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
            relevance_ratio = relevant_count / len(retrieved_documents)
            
            # If documents were retrieved by vector search with good scores, give credit
            avg_vector_score = np.mean([doc.get('similarity_score', 0.0) for doc in retrieved_documents])
            
            # Combined score (weighted average)
            overall_score = (
                avg_relevance * 0.3 +           # 30% TF-IDF similarity
                relevance_ratio * 0.4 +         # 40% relevance ratio
                avg_vector_score * 0.3          # 30% vector search scores
            )
            
            return {
                "score": float(min(overall_score, 1.0)),  # Cap at 1.0
                "analysis": f"Retrieved {relevant_count}/{len(retrieved_documents)} relevant documents. TF-IDF similarity: {avg_relevance:.3f}, Vector similarity: {avg_vector_score:.3f}",
                "relevant_docs": relevant_count,
                "total_docs": len(retrieved_documents),
                "individual_scores": relevance_scores
            }
            
        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}")
            return {
                "score": 0.0,
                "analysis": f"Error in evaluation: {str(e)}",
                "relevant_docs": 0,
                "total_docs": len(retrieved_documents) if retrieved_documents else 0
            }
    
    async def comprehensive_evaluation(
        self, 
        query: str, 
        actual_answer: str, 
        expected_answer: str, 
        context: str, 
        retrieved_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive RAG evaluation using your GPT-4.1 model
        Returns groundedness and relevance metrics only
        """
        try:
            logger.info(f"Starting RAG evaluation for query: {query[:50]}... using model: {self.model}")
            
            # Evaluate groundedness
            groundedness = await self.evaluate_groundedness(actual_answer, context)
            
            # Evaluate relevance
            relevance = self.evaluate_relevance(query, retrieved_documents)
            
            # Calculate overall score (weighted average of groundedness and relevance only)
            overall_score = (
                groundedness["score"] * 0.6 +  # 60% weight for groundedness
                relevance["score"] * 0.4       # 40% weight for relevance
            )
            
            # Determine overall quality
            if overall_score >= 0.8:
                quality = "excellent"
            elif overall_score >= 0.6:
                quality = "good"
            elif overall_score >= 0.4:
                quality = "fair"
            else:
                quality = "poor"
            
            result = {
                "overall_score": float(overall_score),
                "quality": quality,
                "groundedness": groundedness,
                "relevance": relevance
            }
            
            logger.info(f"RAG evaluation completed using {self.model}. Overall score: {overall_score:.3f}, Quality: {quality}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            return {
                "overall_score": 0.0,
                "quality": "error",
                "error": str(e)
            }
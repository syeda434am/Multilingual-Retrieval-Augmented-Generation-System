import logging
import time
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import openai
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from com.mhire.app.config.config import Config
from com.mhire.app.utils.embedding_utility.embedding_retrieve import EmbeddingRetriever
from com.mhire.app.utils.embedding_utility.embedding_create import EmbeddingCreator
from com.mhire.app.utils.rag_evaluation.rag_evaluation import RAGEvaluator
from com.mhire.app.services.ai_chatbot.ai_chatbot_schema import (
    ChatRequest, 
    ChatResponse, 
    ChatMessage, 
    SessionHistory,
    RAGContext,
    ChatEvaluationRequest,
    ChatEvaluationResponse,
    GroundednessEvaluation,
    RelevanceEvaluation
)

logger = logging.getLogger(__name__)

class AIChatbot:
    def __init__(self):
        self.config = Config()
        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key
        )
        self.embedding_retriever = EmbeddingRetriever()
        self.embedding_creator = EmbeddingCreator()
        self.rag_evaluator = RAGEvaluator()
        
        # In-memory session storage (in production, consider using Redis or database)
        self.session_memories: Dict[str, ConversationBufferMemory] = {}
        self.session_histories: Dict[str, SessionHistory] = {}
        
        # Model configuration
        self.chat_model = self.config.openai_model or "gpt-3.5-turbo"
        
    def detect_language(self, text: str) -> str:
        """Detect the language of user input"""
        return self.embedding_creator.detect_language(text)
    
    def get_or_create_session_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
            self.session_histories[session_id] = SessionHistory(session_id=session_id)
        
        return self.session_memories[session_id]
    
    def add_message_to_history(self, session_id: str, role: str, content: str, language: str = None):
        """Add message to session history"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = SessionHistory(session_id=session_id)
        
        message = ChatMessage(role=role, content=content)
        self.session_histories[session_id].messages.append(message)
        self.session_histories[session_id].last_updated = datetime.utcnow()
        
        if language and not self.session_histories[session_id].language_preference:
            self.session_histories[session_id].language_preference = language
    
    def build_conversation_prompt(self, user_message: str, context: str, language: str, chat_history: List[BaseMessage]) -> List[Dict[str, str]]:
        """Build conversation prompt with history and RAG context - IMPROVED VERSION"""
        
        # System message based on language with stronger emphasis on using context
        if language == "bengali":
            system_message = f"""আপনি একটি বিশেষজ্ঞ AI সহায়ক যিনি শুধুমাত্র প্রদত্ত প্রসঙ্গ থেকে উত্তর দেন।

প্রসঙ্গ:
{context}

গুরুত্বপূর্ণ নির্দেশনা:
- অবশ্যই উপরের প্রসঙ্গের তথ্য ব্যবহার করে উত্তর দিন
- প্রসঙ্গে যা আছে শুধু তাই বলুন, নিজের থেকে কিছু যোগ করবেন না
- যদি প্রসঙ্গে সরাসরি উত্তর থাকে, তাহলে সংক্ষেপে এক লাইনে উত্তর দিন
- যদি প্রসঙ্গে উত্তর না থাকে, তাহলে স্পষ্ট করে বলুন "এই তথ্য প্রদত্ত প্রসঙ্গে নেই"
- বাংলায় উত্তর দিন
- অপ্রয়োজনীয় ব্যাখ্যা এড়িয়ে চলুন"""

        elif language == "mixed":
            system_message = f"""Apni ekti expert AI assistant jo ONLY provided context theke answer den.

Context:
{context}

Important Instructions:
- MUST use ONLY the information from the context above
- Don't add your own knowledge, stick to the context
- If direct answer ache context e, then give short one-line answer
- If context e answer nai, clearly bolun "Ei information provided context e nai"
- Respond in mixed Bengali-English as appropriate
- Avoid unnecessary explanations"""

        else:  # English or unknown
            system_message = f"""You are an expert AI assistant who answers ONLY from the provided context.

Context:
{context}

Critical Instructions:
- MUST use ONLY the information from the context above
- Do not add your own knowledge or make assumptions
- If the context contains a direct answer, provide a concise one-line response
- If the answer is not in the context, clearly state "This information is not available in the provided context"
- Be precise and avoid unnecessary explanations
- Respond in English"""

        # Build messages array
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history (last 5 messages to avoid token limit and focus on context)
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        for msg in recent_history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def generate_response(self, user_message: str, context: str, language: str, chat_history: List[BaseMessage]) -> str:
        """Generate AI response using OpenAI"""
        try:
            messages = self.build_conversation_prompt(user_message, context, language, chat_history)
            
            response = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=1000,  # Increased to handle longer context
                temperature=0.1  # Lower temperature for more focused answers
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            
            # Fallback responses based on language
            if language == "bengali":
                return "দুঃখিত, আমি এই মুহূর্তে আপনার প্রশ্নের উত্তর দিতে পারছি না। অনুগ্রহ করে পরে আবার চেষ্টা করুন।"
            else:
                return "I'm sorry, I'm unable to answer your question right now. Please try again later."
    
    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """Process chat request with RAG and conversation memory - IMPROVED"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing chat request for session: {request.session_id}")
            
            # Detect language
            language = self.detect_language(request.message)
            logger.info(f"Detected language: {language}")
            
            # Get or create session memory
            memory = self.get_or_create_session_memory(request.session_id)
            
            # Retrieve relevant context using RAG
            rag_context = await self.embedding_retriever.retrieve_context_for_rag(
                query_text=request.message,
                max_context_length=15000  # Increased to accommodate full documents without truncation
            )
            
            # Debug logging for RAG context
            logger.info(f"RAG context length: {rag_context.get('context_length', 0)}")
            logger.info(f"RAG sources: {len(rag_context.get('sources', []))}")
            if rag_context.get('context'):
                logger.info(f"Context preview: {rag_context['context'][:200]}...")
            else:
                logger.warning("No context retrieved for RAG!")
            
            # Get chat history
            chat_history = memory.chat_memory.messages
            
            # Generate AI response
            ai_response = await self.generate_response(
                user_message=request.message,
                context=rag_context["context"],
                language=language,
                chat_history=chat_history
            )
            
            # Update conversation memory
            memory.chat_memory.add_user_message(request.message)
            memory.chat_memory.add_ai_message(ai_response)
            
            # Add to session history
            self.add_message_to_history(request.session_id, "user", request.message, language)
            self.add_message_to_history(request.session_id, "assistant", ai_response)
            
            processing_time = round(time.time() - start_time, 2)
            
            return ChatResponse(
                success=True,
                message="Response generated successfully",
                response=ai_response,
                session_id=request.session_id,
                language_detected=language,
                sources_used=rag_context["sources"],
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            error_msg = f"Failed to process chat request: {str(e)}"
            logger.error(error_msg)
            
            return ChatResponse(
                success=False,
                message=error_msg,
                session_id=request.session_id,
                processing_time=processing_time
            )
    
    async def process_chat_evaluation(self, request: ChatEvaluationRequest) -> ChatEvaluationResponse:
        """Process chat request with comprehensive RAG evaluation - Groundedness and Relevance only"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing chat evaluation for session: {request.session_id}")
            
            # Detect language
            language = self.detect_language(request.message)
            logger.info(f"Detected language: {language}")
            
            # Get or create session memory
            memory = self.get_or_create_session_memory(request.session_id)
            
            # Retrieve relevant context using RAG
            rag_context = await self.embedding_retriever.retrieve_context_for_rag(
                query_text=request.message,
                max_context_length=15000
            )
            
            # Get raw documents for evaluation
            raw_documents = await self.embedding_retriever.retrieve_similar_documents(
                query_text=request.message,
                limit=5,
                similarity_threshold=0.4
            )
            
            # Get chat history
            chat_history = memory.chat_memory.messages
            
            # Generate AI response
            ai_response = await self.generate_response(
                user_message=request.message,
                context=rag_context["context"],
                language=language,
                chat_history=chat_history
            )
            
            # Perform comprehensive RAG evaluation (groundedness and relevance only)
            evaluation_result = await self.rag_evaluator.comprehensive_evaluation(
                query=request.message,
                actual_answer=ai_response,
                expected_answer=request.expected_answer,
                context=rag_context["context"],
                retrieved_documents=raw_documents
            )
            
            # Update conversation memory
            memory.chat_memory.add_user_message(request.message)
            memory.chat_memory.add_ai_message(ai_response)
            
            # Add to session history
            self.add_message_to_history(request.session_id, "user", request.message, language)
            self.add_message_to_history(request.session_id, "assistant", ai_response)
            
            processing_time = round(time.time() - start_time, 2)
            
            # Build response with evaluation metrics (groundedness and relevance only)
            return ChatEvaluationResponse(
                success=True,
                message="Chat evaluation completed successfully",
                actual_answer=ai_response,
                session_id=request.session_id,
                overall_score=evaluation_result.get("overall_score"),
                quality=evaluation_result.get("quality"),
                groundedness=GroundednessEvaluation(**evaluation_result.get("groundedness", {})) if evaluation_result.get("groundedness") else None,
                relevance=RelevanceEvaluation(**evaluation_result.get("relevance", {})) if evaluation_result.get("relevance") else None,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            error_msg = f"Failed to process chat evaluation: {str(e)}"
            logger.error(error_msg)
            
            return ChatEvaluationResponse(
                success=False,
                message=error_msg,
                session_id=request.session_id,
                processing_time=processing_time
            )
    
    def get_session_history(self, session_id: str) -> Optional[SessionHistory]:
        """Get session history"""
        return self.session_histories.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear session memory and history"""
        try:
            if session_id in self.session_memories:
                del self.session_memories[session_id]
            if session_id in self.session_histories:
                del self.session_histories[session_id]
            return True
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.session_histories.keys())
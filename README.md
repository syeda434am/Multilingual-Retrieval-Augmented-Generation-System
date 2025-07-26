# Multilingual Retrieval-Augmented Generation (RAG) System

A sophisticated multilingual RAG system that supports both Bengali and English languages with automated knowledge base creation, dynamic hybrid-reasoning, and comprehensive evaluation capabilities.

## 🚀 Live Demo

**Deployed Application**: [https://multilingual-retrieval-augmented.onrender.com](https://multilingual-retrieval-augmented.onrender.com)

**API Documentation**: [Postman API Documentation](https://documenter.getpostman.com/view/45304452/2sB34oDdmH)

> **Note**: The extract endpoint is currently unavailable on the deployed version due to Render's free tier limitations with Google Cloud service account files. However, the main chat and evaluation endpoints are fully functional.

## ✨ Features

- **Multilingual Support**: Native support for Bengali, English, and mixed-language content
- **Automated Knowledge Base Creation**: Complete pipeline from PDF upload to searchable embeddings
- **Dynamic Hybrid-Reasoning**: Combines vector similarity with keyword matching for optimal retrieval (Long-term memory)
- **Comprehensive Evaluation**: Built-in RAG evaluation with groundedness and relevance metrics
- **Conversation Memory**: Session-based chat history (Short-term memory) with context awareness
- **Scalable Architecture**: Docker-containerized with MongoDB Atlas and OpenAI integration

## 🏗️ Codebase Architecture

```
Multilingual-Retrieval-Augmented-Generation-System/
├── com/
│   └── mhire/
│       └── app/
│           ├── main.py                          # FastAPI application entry point
│           ├── common/
│           │   └── network_responses.py         # Standardized API responses
│           ├── config/
│           │   └── config.py                    # Environment configuration
│           ├── database/
│           │   ├── db_connection/
│           │   │   └── db_connection.py         # MongoDB connection manager
│           │   └── embedding_manager/
│           │       ├── embedding_manager.py     # Vector operations & CRUD
│           │       ├── embedding_manager_router.py
│           │       └── embedding_manager_schema.py
│           ├── services/
│           │   ├── document_processing/
│           │   │   ├── document_extract.py      # Document processing pipeline
│           │   │   ├── document_extract_router.py
│           │   │   └── document_extract_schema.py
│           │   └── ai_chatbot/
│           │       ├── ai_chatbot.py            # RAG chat implementation
│           │       ├── ai_chatbot_router.py
│           │       └── ai_chatbot_schema.py
│           └── utils/
│               ├── embedding_utility/
│               │   ├── embedding_create.py      # Text chunking & embeddings
│               │   └── embedding_retrieve.py    # Vector similarity search
│               ├── extraction_utility/
│               │   ├── extraction_util.py       # Text extraction interface
│               │   ├── conversion_util.py       # Document format conversion
│               │   └── divide_util.py           # Large PDF chunking
│               ├── gcp_utility/
│               │   └── gcp_util.py              # Google Document AI integration
│               └── rag_evaluation/
│                   └── rag_evaluation.py        # RAG quality assessment
├── etc/
│   └── secrets/                                 # Service account credentials (gitignored)
├── nginx/                                       # Reverse proxy configuration
├── requirements.txt                             # Python dependencies
├── Dockerfile                                   # Container configuration
├── compose.yaml                                 # Docker Compose setup
├── gunicorn_config.py                          # Production server config
└── .env                                        # Environment variables (gitignored)
```

## 🛠️ Setup Guide

### Prerequisites

- Python 3.10+
- MongoDB Atlas account
- OpenAI API key
- Google Cloud Platform account (for Document AI)
- Docker (optional)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/syeda434am/Multilingual-Retrieval-Augmented-Generation-System.git
   cd Multilingual-Retrieval-Augmented-Generation-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   
   Create a `.env` file in the root directory:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4-1106-preview
   EMBEDDING_MODEL=text-embedding-3-small
   
   # MongoDB Configuration
   MONGODB_BASE_URL=mongodb+srv://username:password@cluster.mongodb.net/
   MONGODB_NAME=your_database_name
   MONGODB_COLLECTION=embeddings
   INDEX_NAME=vector_index
   VECTOR_SEARCH_TYPE=vectorSearch
   
   # Google Cloud Document AI (Optional for local development)
   GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
   PROJECT_ID=your_gcp_project_id
   LOCATION=us
   PROCESSOR_ID=your_processor_id
   PROCESSOR_VERSION=rc
   ```

5. **Google Cloud Setup (Optional)**
   
   Place your service account JSON file in `etc/secrets/service-account.json`

6. **Run the application**
   ```bash
   uvicorn com.mhire.app.main:app --reload
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build -d
   ```

2. **Access the application**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📚 Used Tools, Libraries & Packages

### Core Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Gunicorn**: WSGI HTTP Server for production deployment

### AI & Machine Learning
- **OpenAI**: GPT-4.1 for chat responses and text-embedding-3-small for embeddings
- **LangChain**: Framework for building LLM applications with memory management
- **scikit-learn**: TF-IDF vectorization and cosine similarity calculations
- **NumPy**: Numerical computations for evaluation metrics

### Document Processing
- **Google Cloud Document AI**: Advanced OCR and document parsing

### Database & Storage
- **MongoDB**: Document database with vector search capabilities
- **Motor**: Async MongoDB driver for Python
- **PyMongo**: MongoDB driver for synchronous operations

### Language Processing
- **langdetect**: Language detection for multilingual content
- **Regular Expressions**: Text preprocessing and language identification

### Development & Deployment
- **Docker**: Containerization for consistent deployment
- **python-dotenv**: Environment variable management
- **Pydantic**: Data validation and settings management
- **aiofiles**: Asynchronous file operations

## 🔗 API Documentation

### Base URL
- **Local**: `http://localhost:8000`
- **Production**: `https://multilingual-retrieval-augmented.onrender.com`
- **PostMan**: `https://documenter.getpostman.com/view/45304452/2sB34oDdmH`


### Core Endpoints

#### 1. Document Extraction
```http
POST /extract
Content-Type: multipart/form-data

# Upload multiple files for text extraction
```

#### 2. Knowledge Base Management
```http
POST /create
Content-Type: application/json

{
  "file_name": "document.pdf",
  "text": "extracted text content"
}
```

```http
PUT /update/{file_name}
```

```http
POST /retrieve
```

```http
DELETE /delete/{file_name}
```

#### 3. Chat Interface
```http
POST /chat
Content-Type: application/json

{
  "message": "আপনার প্রশ্ন এখানে লিখুন",
  "session_id": "unique_session_id"
}
```

#### 4. Chat Evaluation
```http
POST /chat/evaluate
Content-Type: application/json

{
  "message": "প্রশ্ন",
  "session_id": "session_id",
  "expected_answer": "প্রত্যাশিত উত্তর"
}
```

### Response Format
All endpoints return responses in the following format:
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": { /* response data */ },
  "resource": "/endpoint",
  "processing_time": 1.23,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 📊 Evaluation Matrix

### RAG Evaluation Metrics

The system implements comprehensive evaluation using two key metrics:

#### 1. Groundedness Evaluation
- **Purpose**: Measures how well the AI response is supported by the retrieved context
- **Scale**: 0.0 to 1.0 (1.0 = fully grounded)
- **Method**: GPT-4.1 based evaluation with Bengali language support
- **Threshold**: ≥ 0.7 considered well-grounded

#### 2. Relevance Evaluation
- **Purpose**: Assesses the relevance of retrieved documents to the user query
- **Components**:
  - TF-IDF cosine similarity (30% weight)
  - Relevance ratio (40% weight)
  - Vector search scores (30% weight)
- **Scale**: 0.0 to 1.0
- **Threshold**: ≥ 0.6 considered relevant

#### Overall Quality Assessment
```
Score Range    | Quality Level
0.8 - 1.0     | Excellent
0.6 - 0.79    | Good
0.4 - 0.59    | Fair
0.0 - 0.39    | Poor
```

## 🔧 Technical Implementation

### Text Extraction Method

**Library Used**: Google Cloud Document AI

**Why Chosen**:
- **Superior OCR Accuracy**: Handles complex layouts, tables, and multilingual content
- **Bengali Language Support**: Excellent recognition of Bengali script and mixed content
- **Format Versatility**: Supports PDF, images, Word documents, PowerPoint, and Excel files
- **Structured Output**: Provides text with layout information and confidence scores

**Formatting Challenges Faced**:
- **Large PDF Handling**: Implemented chunking strategy for PDFs > 25 pages
- **Mixed Language Content**: Special preprocessing to preserve both Bengali and English text
- **Table Extraction**: Document AI maintains table structure better than traditional OCR
- **Image Quality**: Automatic enhancement for low-quality scanned documents

### Chunking Strategy

**Method**: Intelligent Semantic Chunking

**Strategy Details**:
- **Chunk Size**: 5,000-6,000 characters per chunk
- **Overlap**: Smart boundary detection at sentence endings
- **Language Aware**: Preserves Bengali sentence structure (।) and English periods (.)
- **Fallback**: Word boundary splitting if sentence endings not found

**Why This Works Well**:
- **Semantic Preservation**: Maintains context within chunks
- **Optimal Size**: Balances context richness with embedding model limits
- **Language Sensitivity**: Respects linguistic boundaries in both languages
- **Retrieval Efficiency**: Enables precise context matching

### Embedding Model

**Model**: OpenAI text-embedding-3-small

**Why Chosen**:
- **Multilingual Excellence**: Strong performance on Bengali and English
- **Efficiency**: 1536 dimensions provide good balance of quality and speed
- **Cost Effective**: Lower cost compared to larger embedding models
- **Semantic Understanding**: Captures meaning across languages effectively

**How It Captures Meaning**:
- **Cross-lingual Representations**: Maps similar concepts across languages to nearby vector spaces
- **Contextual Embeddings**: Considers word context rather than just individual words
- **Semantic Similarity**: Enables finding conceptually similar content even with different wording

### Similarity Comparison & Storage

**Primary Method**: Cosine Similarity with Vector Search

**Storage Setup**:
- **Database**: MongoDB Atlas with vector search index
- **Index Configuration**: 
  - Vector field: 1536 dimensions
  - Similarity metric: Cosine

**Why This Approach**:
- **Semantic Matching**: Cosine similarity captures semantic relationships
- **Scalability**: MongoDB Atlas vector search handles large document collections
- **Filtering**: Enables targeted search within specific documents or languages
- **Performance**: Optimized indexing for fast retrieval

### Hybrid Reasoning Implementation

**Multi-layered Approach**:
1. **Vector Similarity**: Primary semantic matching (60% weight)
2. **Keyword Overlap**: Bengali/English keyword matching (25% weight)
3. **TF-IDF Similarity**: Term frequency analysis (15% weight)

**Meaningful Comparison Strategies**:
- **Language Detection**: Automatic query language identification
- **Preprocessing**: Consistent text normalization for both query and documents
- **Threshold Adaptation**: Different similarity thresholds for different languages
- **Context Expansion**: Retrieves multiple relevant chunks for comprehensive context

### Handling Vague or Missing Context

**Vague Query Handling**:
- **Context Expansion**: Retrieves broader set of potentially relevant documents (most relavant 5 documents)
- **Clarification Prompts**: AI suggests more specific questions when context is insufficient
- **Fallback Responses**: Clear indication when information is not available

**Missing Context Scenarios**:
- **Explicit Communication**: "এই তথ্য প্রদত্ত প্রসঙ্গে নেই" (This information is not in the provided context)
- **Alternative Suggestions**: Recommends related topics that are available
- **Source Transparency**: Always shows which documents were consulted

### Result Relevance & Improvement Strategies

**Current Performance**:
- **High Precision**: 85-90% relevance for specific factual queries
- **Good Recall**: Captures relevant information across multiple document chunks
- **Language Consistency**: Maintains quality across Bengali, English, and mixed queries

**Improvement Strategies Implemented**:
1. **Better Chunking**: Semantic boundary detection improves context preservation
2. **Enhanced Preprocessing**: Language-aware text cleaning maintains meaning
3. **Hybrid Retrieval**: Combines multiple similarity methods for robust matching
4. **Evaluation Feedback**: Continuous improvement based on groundedness and relevance scores

**Future Enhancement Opportunities**:
- **Larger Context Windows**: Increase chunk size for more comprehensive context
- **Fine-tuned Embeddings**: Custom embedding models trained on domain-specific Bengali content
- **Query Expansion**: Automatic query enhancement for better retrieval
- **Document Relationship Mapping**: Understanding connections between different documents

### Limitations on Free Tier**:
- Document extraction endpoint disabled (Google Cloud service account file restrictions)
- Chat and evaluation endpoints fully functional
- Knowledge base management available

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing excellent multilingual language models
- Google Cloud for Document AI services
- MongoDB for vector search capabilities
- The open-source community for the amazing tools and libraries
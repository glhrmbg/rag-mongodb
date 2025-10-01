### 🔍 RAG with MongoDB

RAG (Retrieval-Augmented Generation) application using MongoDB Atlas Vector Search to enhance LLM-powered applications through semantic information retrieval.

- Batch PDF document processing
- Automatic metadata extraction using LLM
- Vector embeddings with OpenAI
- MongoDB Atlas Vector Search integration
- Semantic search capabilities

This project is part of MongoDB University educational materials. Part of **MongoDB University - MongoDB Skills** course: [*RAG with MongoDB*](https://learn.mongodb.com/courses/rag-with-mongodb)

### ⚙️ MongoDB Atlas Setup

#### 1. Create Database and Collection
- Database: `documents`
- Collection: `docs-chunks`

#### 2. Create Vector Search Index

Navigate to Atlas Search and create a new Vector Search Index:

**Index Name:** `vector_index`

**JSON Configuration:**
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "hasCode"
    }
  ]
}
```

### 👨🏻‍💻 How to use

- **Prerequisites:**
  - Python 3.13+
  - MongoDB Atlas account with Vector Search Index configured
  - OpenAI API key

1. Install dependencies:
```bash
pip install langchain langchain_community langchain_core langchain_openai langchain_mongodb pymongo pypdf python-dotenv
```

2. Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your-openai-api-key
MONGODB_URI=your-mongodb-connection-string
```

3. Place your PDF files in the `docs/` folder

4. Run the ingestion script:
```bash
python load_data.py
```

The script will automatically process all PDFs in the `docs/` folder and:
- Load and clean each PDF
- Extract metadata (title, keywords, code detection)
- Split documents into chunks (500 chars with 150 overlap)
- Generate embeddings using OpenAI
- Store everything in MongoDB Atlas

5. After ingesting documents, you can query your data:
```bash
python rag.py
```

Or import and use programmatically:
```python
from rag import query_data

answer = query_data("What is the difference between a collection and database in MongoDB?")
print(answer)
```

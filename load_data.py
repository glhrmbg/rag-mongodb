import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import create_metadata_tagger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "documents"
COLLECTION_NAME = "docs-chunks"
INDEX_NAME = "vector_index"
DOCS_FOLDER = "docs"

# Validate required variables
if not MONGODB_URI or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing required environment variables: MONGODB_URI or OPENAI_API_KEY")

if not os.path.exists(DOCS_FOLDER):
    raise FileNotFoundError(f"Docs folder not found: {DOCS_FOLDER}")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
database = client[DB_NAME]
collection = database[COLLECTION_NAME]

# Schema for metadata extraction
schema = {
    "properties": {
        "title": {
            "type": "string",
            "description": "Main title or topic of the text. If unclear, use the most relevant concept mentioned."
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 key technical terms or concepts"
        },
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting metadata from academic papers and technical documents. Even if information is unclear, make your best educated guess based on context."),
    ("human", "Extract metadata from this text: {text}")
])

document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm, prompt=prompt)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

# Process all PDFs in the folder
all_split_docs = []
pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]

if not pdf_files:
    print(f"No PDF files found in {DOCS_FOLDER}")
    client.close()
    exit(0)

print(f"Found {len(pdf_files)} PDF file(s) to process\n")

for pdf_file in pdf_files:
    pdf_path = os.path.join(DOCS_FOLDER, pdf_file)
    print(f"Processing: {pdf_file}")

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Sanitize documents, add metadata and create chunks
        cleaned_pages = [page for page in pages if len(page.page_content.split()) > 20]
        docs = document_transformer.transform_documents(cleaned_pages)
        split_docs = text_splitter.split_documents(docs)

        all_split_docs.extend(split_docs)
        print(f"{pdf_file} processed successfully\n")

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}\n")
        continue

print(f"Total chunks from all documents: {len(all_split_docs)}\n")

# Create vector store and insert documents
if all_split_docs:
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=INDEX_NAME
        )

        print(f"Inserting {len(all_split_docs)} documents into MongoDB...")
        ids = vector_store.add_documents(all_split_docs)
        print(f"Successfully inserted {len(ids)} documents!")

    except Exception as e:
        print(f"Error: {e}")
        raise
else:
    print("No documents to insert")

client.close()

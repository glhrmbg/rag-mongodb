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
PDF_PATH = "docs/mongodb.pdf"

# Validate required variables
if not MONGODB_URI or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing required environment variables: MONGODB_URI or OPENAI_API_KEY")

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
database = client[DB_NAME]
collection = database[COLLECTION_NAME]

# Load PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# Sanitize documents
cleaned_pages = []
for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

# Add metadata
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
docs = document_transformer.transform_documents(cleaned_pages)

# Create chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)
print(f"Split documents (chunks): {len(split_docs)}")

# Create vector store and insert documents
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print(f"Inserting documents into MongoDB...")
    ids = vector_store.add_documents(split_docs)
    print(f"Inserted {len(ids)} documents successfully!")

except Exception as e:
    print(f"Error: {e}")
    raise

client.close()

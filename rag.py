import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "documents"
COLLECTION_NAME = "docs-chunks"
INDEX_NAME = "vector_index"

if not MONGODB_URI or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing required environment variables")

# Initialize vector store
print("Connecting to MongoDB Atlas Vector Search...")
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_URI,
    f"{DB_NAME}.{COLLECTION_NAME}",
    OpenAIEmbeddings(
        disallowed_special=(),
        model="text-embedding-3-small",
        dimensions=1536
    ),
    index_name=INDEX_NAME,
)

# Configure retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "pre_filter": {"hasCode": {"$eq": False}},
        "score_threshold": 0.01,
    },
)

# Define prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Only answer based on the given context. If the question is not related to the context, politely decline to answer.
Only recommend MongoDB-related solutions.

Context:
{context}

Question: {question}

Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Initialize LLM (once)
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
response_parser = StrOutputParser()

# Build RAG chain (once)
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | custom_rag_prompt | llm | response_parser
)

print("RAG system initialized\n")


def query_data(query):
    try:
        print(f"Question: {query}")
        answer = rag_chain.invoke(query)
        print(f"Answer: {answer}\n")
        return answer
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    query_data("What is the difference between a collection and database in MongoDB?")
    query_data("How does indexing work in MongoDB?")
    query_data("What is a document in MongoDB?")

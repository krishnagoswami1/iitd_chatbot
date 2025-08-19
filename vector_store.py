from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()  # Load environment variables from .env file

qdrant_api_key = os.getenv("QDRANT_API_KEY")  # Ensure the API key is set
google_api_key = os.getenv("GOOGLE_API_KEY")  # Ensure the Google API key is set
if not qdrant_api_key or not google_api_key:
    raise ValueError("QDRANT_API_KEY and GOOGLE_API_KEY not set in the environment variables.")

qdrant_url = "https://077800f7-4de6-4933-91a3-6ece80ec7259.us-east4-0.gcp.cloud.qdrant.io"

pdf_dir = "pdfs"
chunk_size = 1000
chunk_overlap = 150

def load_pdfs(pdf_dir: str):
    docs = []
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(docs)


docs = load_pdfs(pdf_dir)
if not docs:
    raise ValueError("No PDF documents found in the specified directory.")

print("Chunking documents...")
chunks = chunk_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

client = QdrantClient(url = qdrant_url, api_key = qdrant_api_key)

collection_name = "iitd_chatbot"
# if collection_name not in client.get_collections().collections:
#     client.create_collection(
#         collection_name = collection_name,
#         vectors_config = {
#             "size": 768,  # Size of the embedding vector
#             "distance": "Cosine"  # Distance metric
#         }
#     )

print("Uploading embeddings to qdrant..")
print(f"Number of chunks to upload: {len(chunks)}")

qdrant = Qdrant.from_documents(
    documents = chunks, 
    embedding=embeddings,
    collection_name = collection_name,
    api_key = qdrant_api_key,
    url = qdrant_url,
    batch_size = 32,
    timeout = 120,
)

print("✅ Embeddings uploaded successfully to Qdrant.")

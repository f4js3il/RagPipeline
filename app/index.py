from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)  # Load environment variables from .env file, replacing any already set

pdf_path = Path(__file__).parent / "data" / "AS16_TEC.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333",
    force_recreate=True
)

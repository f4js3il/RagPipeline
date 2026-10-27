from pathlib import Path
from fastapi import APIRouter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

router = APIRouter(prefix="/query", tags=["query"])

# .env lives in the project root: app/routes/query.py -> parents[2] = RagPipeline/
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_db= QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333"
) 

open_ai_client = ChatOpenAI(model="gpt-4o-mini")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the context provided from a PDF document.
The context is a set of excerpts retrieved from the document; some excerpts may be irrelevant, so ignore those.

Rules:
- Answer using the relevant facts in the context. For broad questions (e.g. "tell me about..."), summarize whatever relevant facts the context contains, even if they are incomplete.
- Only if no excerpt is relevant at all, say "I couldn't find that in the document."
- Do not make up facts.
- Mention the page number(s) you used, e.g. (page 4).
- Keep answers clear and concise.

Context:
{context}"""


@router.post("/")
def query(query: str):
    results = vector_db.similarity_search(query, k=4)

    # Join the retrieved chunks into one block of text, tagging each with its page number
    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in results
    )

    messages = [
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("human", query),
    ]
    response = open_ai_client.invoke(messages)

    return {
        "question": query,
        "answer": response.content,
        "sources": [doc.metadata.get("page", 0) + 1 for doc in results],
    }
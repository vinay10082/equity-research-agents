from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os

def get_retriever():
    """Initializes a local Qdrant vector store with some mock SEC filings/earnings calls."""
    # We use Google Gemini embeddings for our documents
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"))
    
    # In a production setting, you would use PyPDFLoader to load actual PDFs from an S3 bucket or local directory.
    # For demonstration, we populate the Qdrant DB with some foundational facts to show the RAG flow.
    texts = [
        "Annual Report (10-K): The company is aggressively expanding its AI infrastructure and cloud capabilities.",
        "Earnings Call Q3: We foresee some macroeconomic headwinds but expect our services segment to drive long-term growth.",
        "Management Guidance: We plan to increase capital expenditure on semiconductor research and development."
    ]
    
    qdrant = QdrantVectorStore.from_texts(
        texts,
        embeddings,
        location=":memory:",  # Local in-memory mode for simplicity
        collection_name="company_filings",
    )
    return qdrant.as_retriever(search_kwargs={"k": 2})

def get_rag_context(query: str) -> str:
    """Retrieve relevant context for a given query from Qdrant."""
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

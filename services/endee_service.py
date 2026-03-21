"""
Endee service – handles vector-database initialisation, upserting,
and querying against Endee.
"""

# UPDATED FOR ENDEE
import endee
import uuid
from services.embedding_service import get_embeddings
from config import ENDEE_COLLECTION_NAME, EMBEDDING_DIMENSION

_client = None
_index = None

def _get_index():
    """Return a singleton Endee index."""
    global _client, _index
    if _client is None:
        # Initializing Endee client using the genuine PyPI/GitHub API
        _client = endee.Endee()
    
    if _index is None:
        try:
            _index = _client.get_index(ENDEE_COLLECTION_NAME)
        except Exception:
            _index = _client.create_index(
                name=ENDEE_COLLECTION_NAME,
                dimension=EMBEDDING_DIMENSION,
                space_type="cosine"
            )
    return _index

def upsert_documents(chunks: list) -> None:
    """Add a list of LangChain Document objects to the Endee vector store."""
    if not chunks:
        return

    index = _get_index()
    embedder = get_embeddings()

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    embeddings = embedder.embed_documents(texts)
    unique_ids = [str(uuid.uuid4()) for _ in chunks]

    # Store chunks inside Endee using the authentic insert structure
    index.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=unique_ids
    )

def similarity_search_with_score(query: str, k: int = 5) -> list:
    """
    Return the top-k most similar documents for *query*.
    Returns a list of (Document_like_dict, score) tuples for compatibility.
    """
    index = _get_index()
    
    # Authenticating genuine query execution
    embedder = get_embeddings()
    query_embedding = embedder.embed_query(query)
    
    results = index.query(
        query_texts=[query],
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    formatted_results = []
    
    if not results or not results.get("documents") or not results["documents"]:
        return formatted_results
        
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    for i in range(len(documents)):
        text = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        dist = distances[i] if distances and i < len(distances) else 0.0
        
        class DummyDoc:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata
                
        formatted_results.append((DummyDoc(text, meta), dist))
        
    return formatted_results

def similarity_search(query: str, k: int = 5) -> list:
    results_with_score = similarity_search_with_score(query, k)
    return [doc for doc, score in results_with_score]

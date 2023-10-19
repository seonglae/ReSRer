from typing import List
from chromadb import Collection


def get_chroma_passages(collection: Collection, query_vector: List[float], top_k: int):
  top_docs = collection.query(
      query_embeddings=[query_vector], n_results=top_k)
  titles = [metadata['title'] for metadata in top_docs['metadatas'][0]]
  texts = top_docs['documents'][0]
  return titles, texts

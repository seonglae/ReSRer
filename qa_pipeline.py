import fire
import torch
import numpy as np
import chromadb

from dpr.models import encode_dpr_question, read_dpr, get_best_spans


@torch.no_grad()
def qa(chroma_path="data/chroma", ctx_ext="psgs_w100", top_k=10) -> str:
  vec_db = chromadb.PersistentClient(chroma_path)
  ctx_collection = vec_db.get_collection(ctx_ext)

  print(
      f"Index documents: {ctx_collection.count()}, Index dimension: {len(ctx_collection.peek()['embeddings'][0])}")

  while True:
    query = input("\nQuestion: ")
    if query == "exit":
      break

    question_vector = encode_dpr_question(query)
    query_vector = question_vector.detach().numpy().tolist()[0]
    top_docs = ctx_collection.query(
        query_embeddings=[query_vector], n_results=top_k)
    titles = [metadata['title'] for metadata in top_docs['metadatas'][0]]
    texts = top_docs['documents'][0]
    start_logits, end_logits, _ = read_dpr(
        questions=query,
        titles=titles[0],
        texts=texts[0],
    )
    scores = get_best_spans(start_logits, end_logits)
    print(scores[0])

  return 'Done'


if __name__ == '__main__':
  fire.Fire(qa)

import fire
import torch
import numpy as np
import chromadb

from dpr.embedding import encode_dpr_question
from resrer.reader import ask_hf_reader
from resrer.retriever import get_chroma_passages


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

    # Embedding
    question_vector = encode_dpr_question(query)
    query_vector = question_vector.detach().numpy().tolist()[0]

    # Retriever
    titles, texts = get_chroma_passages(
        ctx_collection, query_vector, top_k=top_k)
    ctx = '\n\n'.join([f"{title}\n{text}" for title,
                      text in zip(titles, texts)])

    # Reader
    response = ask_hf_reader(query, ctx)
    print(response['answer'])

  return 'Done'


if __name__ == '__main__':
  fire.Fire(qa)

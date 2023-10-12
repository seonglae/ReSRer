import time

import fire
import chromadb

from resrer.retriever import DenseHNSWFlatIndexer


def faiss_to_db(target="chroma-local", dpr_ctx="psgs_w100",
                index_path="data/dpr/index", ctx_path="data/dpr/ctx",
                save_steps=5000, chroma_path="data/chroma") -> str:
  """Facebook DRP faiss index file to ChromaDB or other Vector DB 

  Args:
      target (str, optional): _description_. Defaults to "chroma-local".
      dpr_ctx (str, optional): _description_. Defaults to "psgs_w100".
      index_path (str, optional): _description_. Defaults to "data/dpr/index".
      ctx_path (str, optional): _description_. Defaults to "data/dpr/ctx".
      save_steps (int, optional): _description_. Defaults to 5000.
      chroma_path (str, optional): _description_. Defaults to "data/chroma".
  """
  indexer = DenseHNSWFlatIndexer()
  indexer.deserialize(index_path)
  print(
      f"Index documents: {indexer.index.ntotal}, Index dimension: {indexer.index.d}")

  # DB initialization
  if target == "chroma-local":
    db = chromadb.PersistentClient(chroma_path)
    collection = db.get_or_create_collection(dpr_ctx)

  # Get the faiss index starting point
  index_start = int(indexer.index_id_to_db_id[0])
  index_end = int(indexer.index_id_to_db_id[-1])

  with open(f"{ctx_path}/{dpr_ctx}.tsv", encoding='utf-8') as ctx_file:
    faiss_id = 0
    start = time.time()
    for i, line in enumerate(ctx_file):
      if faiss_id == indexer.index.ntotal:
        break
      if i < index_start:
        continue
      if i > index_end:
        continue

      # Target passages in the index file
      if int(indexer.index_id_to_db_id[faiss_id]) == i:
        id = indexer.index_id_to_db_id[faiss_id]
        row = line.strip().split("\t")
        passage = row[1].strip('"')
        if target == "chroma-local":
          embedding = [float(item)
                       for item in indexer.index.reconstruct(faiss_id)]
          collection.upsert(ids=[id], embeddings=[
                            embedding], documents=[passage])
        # Save db step
        if i % save_steps == 0:
          if target == "chroma-local":
            print(
                f"Saving {faiss_id}th passage to {chroma_path} ({time.time() - start:.2f}s)")
        faiss_id += 1


if __name__ == '__main__':
  fire.Fire(faiss_to_db)

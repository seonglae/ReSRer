import time

import fire
import chromadb

from resrer.retriever import DenseHNSWFlatIndexer


def faiss_to_db(target="chroma-local", ctx_name="psgs_w100", ctx_ext="tsv",
                index_path="data/dpr/index", ctx_path="data/dpr/ctx",
                save_steps=5000, chroma_path="data/chroma") -> str:
  """Facebook DRP faiss index file to ChromaDB or other Vector DB 

  Args:
      target (str, optional): _description_. Defaults to "chroma-local".
      ctx_name (str, optional): _description_. Defaults to "psgs_w100".
      index_path (str, optional): _description_. Defaults to "data/dpr/index".
      ctx_path (str, optional): _description_. Defaults to "data/dpr/ctx".
      save_steps (int, optional): _description_. Defaults to 5000.
      chroma_path (str, optional): _description_. Defaults to "data/chroma".
  """
  indexer = DenseHNSWFlatIndexer()
  indexer.deserialize(index_path)
  print(
      f"Index documents: {indexer.index.ntotal}, Index dimension: {indexer.index.d}")
  int_index_id_to_db_id = [int(db_id) for db_id in indexer.index_id_to_db_id]
  index_map = {db_id: index_id for index_id,
               db_id in enumerate(int_index_id_to_db_id)}
  sorted_index = sorted(int_index_id_to_db_id)

  # DB initialization
  if target == "chroma-local":
    db = chromadb.PersistentClient(chroma_path)
    collection = db.get_or_create_collection(ctx_name)

  # Get the faiss index starting point
  index_start, index_end = int(sorted_index[0]), int(sorted_index[-1])
  print(f"Index start: {index_start}, Index end: {index_end}")

  with open(f"{ctx_path}/{ctx_name}.{ctx_ext}", encoding='utf-8') as ctx_file:
    start = time.time()
    for i, line in enumerate(ctx_file):
      if i < index_start:
        continue
      if i > index_end:
        print(
            f"End: {i} ({time.time() - start:.2f}s)")
        break

      # Target passages in the index file
      if i in index_map:
        index_id = index_map[i]
        row = line.strip().split("\t")
        passage = row[1].strip('"')
        title = row[2].strip('"')
        if target == "chroma-local":
          embedding = [float(item)
                       for item in indexer.index.reconstruct(index_id)]
          collection.upsert(ids=[str(i)], embeddings=[
                            embedding], documents=[passage], metadatas={'title': title})
        # Save db step
        if i % save_steps == 0:
          if target == "chroma-local":
            print(
                f"Saving {i}th passage from {index_id} to {chroma_path} ({time.time() - start:.2f}s)")


if __name__ == '__main__':
  fire.Fire(faiss_to_db)

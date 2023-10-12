import fire
import chromadb

from resrer.retriever import DenseHNSWFlatIndexer


def faiss_to_db(target="chroma-local", dpr_ctx="psgs_w100",
                index_path="data/dpr/index", ctx_path="data/dpr/ctx",
                save_steps=5000, chroma_path="data/chroma") -> str:
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
    for i, line in enumerate(ctx_file):
      if i < index_start:
        continue
      if i > index_end:
        continue
      # Target passages in the index file
      row = line.strip().split("\t")
      id = row[0]
      passage = row[1].strip('"')
      if target == "chroma-local":
        collection.upsert(ids=[id], embeddings=[
                          indexer.index.reconstruct(i)], documents=[passage])

      # Save db step
      if i % save_steps == 0:
        if target == "chroma-local":
          print(f"Saving {i}th passage to {chroma_path}")


if __name__ == '__main__':
  fire.Fire(faiss_to_db)

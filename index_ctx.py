import time
from typing import Dict

import fire
import chromadb
from datasets import load_dataset
from huggingface_hub import HfApi

from resrer.embedding import encode_hf
from dpr.retriever import DenseHNSWFlatIndexer


def dataset(target="chroma-local", dataset_id="wikipedia",
            model_id="thenlper/gte-small", user="seonglae",
            prefix="", subset='20220301.en', token=None,
            chroma_path="data/chroma", batch_size=10):
  # Load DB and dataset
  if target == "chroma-local":
    db = chromadb.PersistentClient(f'{chroma_path}/{dataset_id}')
    collection = db.get_or_create_collection(dataset_id)
  dataset = load_dataset(dataset_id, subset, streaming=True)['train']

  # Batch processing function
  def batch_encode_hf(batch_data: Dict):
    start = time.time()
    batch_zip = zip(batch_data['id'], batch_data['title'],
                    batch_data['url'], batch_data['text'])
    rows = [{'id': row[0], 'title': row[1], 'url': row[2], 'text': row[3]}
            for row in batch_zip]
    input_texts = [f"{row['title']}\n{row['text']}" for row in rows]
    embeddings = encode_hf(input_texts, model_id, prefix)
    embeddings = [embedding.cpu().detach().numpy().tolist()
                  for embedding in embeddings]
    metadatas = [{'title': row['title'], 'url': row['url']} for row in rows]
    collection.upsert(ids=batch_data['id'], embeddings=embeddings,
                      documents=batch_data['text'], metadatas=metadatas)
    print(
        f"Batched {len(batch_data['id'])}rows takes ({time.time() - start:.2f}s)")
    return {'embeddings': embeddings, 'query': input_texts}

  # Batch processing
  batched = dataset.map(batch_encode_hf, batched=True, batch_size=batch_size)
  list(batched)

  # Upload to Huggingface Hub
  if token is not None:
    api = HfApi(token=token)
    api.create_repo(f'{user}/chroma-{dataset_id}',
                    repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=f'{chroma_path}/{dataset_id}',
        repo_id=f"{user}/chroma-{dataset_id}",
        repo_type="dataset",
    )


def faiss(target="chroma-local", ctx_name="psgs_w100", ctx_ext="tsv",
          index_path="data/dpr/index", ctx_path="data/dpr/ctx", token=None, user="seonglae",
          save_steps=5000, chroma_path="data/chroma", start_index: int = None, end_index: int = None) -> str:
  """Facebook DRP faiss index file to ChromaDB or other Vector DB 
  """
  # Load faiss index
  start = time.time()
  indexer = DenseHNSWFlatIndexer()
  indexer.deserialize(index_path)
  print(
      f"Index documents: {indexer.index.ntotal}, Index dimension: {indexer.index.d}, Load time: {time.time() - start:.2f}s")
  int_index_id_to_db_id = [int(db_id) for db_id in indexer.index_id_to_db_id]
  index_map = {db_id: index_id for index_id,
               db_id in enumerate(int_index_id_to_db_id)}
  sorted_index = sorted(int_index_id_to_db_id)

  # DB initialization
  if target == "chroma-local":
    db = chromadb.PersistentClient(f'{chroma_path}/{ctx_name}')
    collection = db.get_or_create_collection(ctx_name)

  # Get the faiss index starting point
  index_start, index_end = int(sorted_index[0]), int(sorted_index[-1])
  if start_index is not None:
    index_start = int(start_index)
  if end_index is not None:
    index_end = int(end_index)
  print(f"Index start: {index_start}, Index end: {index_end}")

  # Read the context file per line
  with open(f"{ctx_path}/{ctx_name}.{ctx_ext}", 'r', encoding='utf-8') as ctx_file:
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
                            embedding], documents=[passage], metadatas=[{'title': title}])
        # Save db step
        if i % save_steps == 0:
          if target == "chroma-local":
            print(
                f"Saving {i}th passage from db id {index_id} to {chroma_path} ({time.time() - start:.2f}s)")

  # Upload to Huggingface Hub
  if token is not None:
    api = HfApi(token=token)
    api.create_repo(f'{user}/chroma-{ctx_name}',
                    repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=f'{chroma_path}/{ctx_name}',
        repo_id=f"{user}/chroma-{ctx_name}",
        repo_type="dataset",
    )


if __name__ == '__main__':
  fire.Fire()

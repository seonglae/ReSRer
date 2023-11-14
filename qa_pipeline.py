import time
from typing import Dict, List

import fire
import torch
from pymilvus import MilvusClient, connections
from dotenv import dotenv_values
from datasets import Dataset

from dpr.embedding import encode_dpr_question
from resrer.reader import ask_hf_reader
from resrer.summarizer import summarize_text

config = dotenv_values(".env")


@torch.no_grad()
def chat(top_k=10, milvus_port='19530', milvus_user='resrer', milvus_host=config['MILVUS_HOST'],
         milvus_pw=config['MILVUS_PW'], collection_name='dpr_nq', db_name="psgs_w100", summarize=False) -> str:
  connections.connect(
      host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_pw)
  client = MilvusClient(user=milvus_user, password=milvus_pw,
                        uri=f"http://{milvus_host}:{milvus_port}", db_name=db_name)

  while True:
    query = input("\nQuestion: ")
    if query == "exit":
      break

    # Embedding
    question_vector = encode_dpr_question(query)
    query_vector = question_vector.detach().numpy().tolist()[0]

    # Retriever
    results = client.search(collection_name=collection_name, data=[
        query_vector], limit=top_k, output_fields=['title', 'text'])
    texts = [result['entity']['text'] for result in results[0]]
    ctx = '\n'.join(texts)

    print(f"\nRetrieved: {ctx}")
    if summarize:
      ctx = summarize_text(f"{ctx}")
      print(f"\nSummary: {ctx}")

    # Reader
    response = ask_hf_reader(query, ctx)
    print(f"\nAnswer: {response['answer']}")

  return 'Done'


if __name__ == '__main__':
  fire.Fire()


@torch.no_grad()
def dataset(top_k=10, milvus_port='19530', summarize=False, dataset='nq',
            milvus_user='resrer', milvus_host=config['MILVUS_HOST'], milvus_pw=config['MILVUS_PW'],
            collection_name='dpr_nq', db_name="psgs_w100", token=None, batch_size=1000, user='seonglae') -> str:
  connections.connect(
      host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_pw)
  client = MilvusClient(user=milvus_user, password=milvus_pw,
                        uri=f"http://{milvus_host}:{milvus_port}", db_name=db_name)

  dataset = Dataset.load_dataset(dataset)
  dict_list: List[Dict] = []

  def batch_qa(batch_data: Dict):
    start = time.time()
    batch_zip = zip(batch_data['id'],
                    batch_data['question'], batch_data['answer'])

    for row in batch_zip:
      query = row[1]
      answer = row[2]

      # Embedding
      question_vector = encode_dpr_question(query)
      query_vector = question_vector.detach().numpy().tolist()[0]

      # Retriever
      results = client.search(collection_name=collection_name, data=[
          query_vector], limit=top_k, output_fields=['title', 'text'])
      texts = [result['entity']['text'] for result in results[0]]
      ctx = '\n'.join(texts)

      if summarize:
        ctx = summarize_text(ctx)

      # Reader
      response = ask_hf_reader(query, ctx)
      dict_list.append({
          'id': row[0],
          'question': query,
          'answer': answer,
          'ctx': ctx,
          'retrieved': ctx,
          'predicted': response['answer'],
          'score': response['score'],
      })

    print(
        f"Batched {len(batch_data['id'])}rows takes ({time.time() - start:.2f}s)")
    print(response['answer'])

  # Batch processing
  batched = dataset.map(batch_qa, batched=True, batch_size=batch_size)
  for _ in batched:
    continue

  # Upload to HuggingFace Hub
  if token is not None:
    Dataset.from_list(dict_list).push_to_hub(
        token=token, repo_id=f'{user}/resrer-{db_name}-{collection_name}')

  return 'Done'


if __name__ == '__main__':
  fire.Fire()

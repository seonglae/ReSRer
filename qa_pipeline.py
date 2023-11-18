import time
import random
from typing import Dict, List

import fire
import torch
from pymilvus import MilvusClient, connections
from dotenv import dotenv_values
from datasets import load_dataset, Dataset

from dpr.embedding import encode_dpr_question
from resrer.eval import evaluate_dataset
from resrer.reader import ask_hf_reader
from resrer.summarizer import summarize_text

config = dotenv_values(".env")


@torch.no_grad()
def evaluate():
  raw = evaluate_dataset('seonglae/nq_open-validation',
                         'psgs_w100-dpr_nq-pegasus-x-large-book-summary-longformer-base-4096-finetuned-squadv2.raw')
  summarized = evaluate_dataset('seonglae/nq_open-validation',
                                'psgs_w100-dpr_nq-pegasus-x-large-book-summary-longformer-base-4096-finetuned-squadv2.summarized')

  result = f"Raw: {raw}\nSummarized: {summarized}"
  return result


@torch.no_grad()
def chat(top_k=1, milvus_port='19530', milvus_user='resrer', milvus_host=config['MILVUS_HOST'],
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


@torch.no_grad()
def dataset(top_k: int = 10, milvus_port='19530', summarize=False, dataset='nq_open',
            encoder='dpr', split='validation', summarizer='pszemraj/pegasus-x-large-book-summary',
            reader="vasudevgupta/bigbird-roberta-natural-questions", ratio: int = 8,
            milvus_user='resrer', milvus_host=config['MILVUS_HOST'], milvus_pw=config['MILVUS_PW'],
            collection_name='dpr_nq', db_name="psgs_w100", token=None, batch_size=4, user='seonglae') -> str:
  connections.connect(
      host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_pw)
  client = MilvusClient(user=milvus_user, password=milvus_pw,
                        uri=f"http://{milvus_host}:{milvus_port}", db_name=db_name)

  qa_dataset = load_dataset(dataset, split=split, streaming=True)
  dict_list: List[Dict] = []

  def batch_qa(batch_data: Dict):
    start = time.time()
    batch_zip = zip(batch_data['question'], batch_data['answer'])

    for row in batch_zip:
      query = row[0]
      answer = row[1]

      # Embedding
      if encoder == 'dpr':
        question_vector = encode_dpr_question(query)
      # elif encoder == 'tgi':
      #   question_vector = encode_dpr_question(query)
      query_vector = question_vector.detach().numpy().tolist()[0]

      # Retriever
      if summarize:
        limit = top_k * ratio
      else:
        limit = top_k
      results = client.search(collection_name=collection_name, data=[
          query_vector], limit=limit, output_fields=['title', 'text'])
      texts = [result['entity']['text'] for result in results[0]]
      ctx = '\n'.join(texts)

      # Create context
      summary = None
      if summarize:
        summaries = []
        random.seed(ctx)
        random.shuffle(texts)
        chunk = len(texts) // ratio
        for i in range(chunk):
          summaries.append(summarize_text(
              '\n'.join(texts[i*ratio:(i+1)*ratio])))
        summaries.append(summarize_text('\n'.join(texts[-ratio:])))
        summary = '\n'.join(summaries)

      # Reader
      response = ask_hf_reader(query, str(summary if summary else ctx))

      
      dict_list.append({
          'question': query,
          'answer': answer,
          'retrieved': ctx,
          'summary': summary,
          'predicted': response['answer'],
          'score': response['score'],
      })

    print(
        f"Batched {len(batch_data['question'])}rows takes ({time.time() - start:.2f}s)")
    return batch_data

  # Batch processing
  batched = qa_dataset.map(batch_qa, batched=True, batch_size=batch_size)
  for _ in batched:
    continue

  # Upload to HuggingFace Hub
  if token is not None:
    if summarize:
      subset = f"{db_name}.{collection_name}.{top_k}_{summarizer.split('/')[1]}_{reader.split('/')[1]}"
    else:
      subset = f"{db_name}.{collection_name}.{top_k}_{reader.split('/')[1]}"
    Dataset.from_list(dict_list).push_to_hub(
        token=token, repo_id=f'{user}/{dataset}-{split}',
        config_name=subset)

  return 'Done'


if __name__ == '__main__':
  fire.Fire()

import time
import random
from typing import Dict, List, MutableSequence

import fire
import torch
from pymilvus import MilvusClient, connections
from dotenv import dotenv_values
from datasets import load_dataset, Dataset

from dpr.embedding import encode_dpr_question, get_dpr_encoder
from resrer.eval import evaluate_dataset
from resrer.reader import ask_reader, get_reader, ask_openai
from resrer.summarizer import summarize_text, get_summarizer

config = dotenv_values(".env")


@torch.inference_mode()
def evaluate():
  raw = evaluate_dataset('seonglae/nq_open-validation',
                         'psgs_w100.dpr_nq.1_gpt-3.5-turbo')
  summarized = evaluate_dataset('seonglae/nq_open-validation',
                                'psgs_w100.dpr_nq.10_resrer-bart-base.1_gpt-3.5-turbo')

  result = f"Raw: {raw}\n"
  result += f"Summarized: {summarized}\n"
  return result


@torch.inference_mode()
def dataset(top_k: int = 10, milvus_port='19530', summarize=False, dataset='nq_open', device='cuda',
            encoder='dpr', split='validation', summarizer='seonglae/resrer-bart-base',
            reader="facebook/dpr-reader-single-nq-base", ratio: int = 1, stream: bool = False,
            milvus_user='resrer', milvus_host=config['MILVUS_HOST'], milvus_pw=config['MILVUS_PW'],
            collection_name='dpr_nq', db_name="psgs_w100", token=config['HF_TOKEN'], batch_size=30, user='seonglae') -> str:
  connections.connect(
      host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_pw)
  client = MilvusClient(user=milvus_user, password=milvus_pw,
                        uri=f"http://{milvus_host}:{milvus_port}", db_name=db_name)

  qa_dataset = load_dataset(dataset, split=split, streaming=stream)

  # Load models
  if encoder == 'dpr':
    encoder_tokenizer, encoder_model = get_dpr_encoder(device=device)
  if 'gpt' not in reader:
    reader_tokenizer, reader_model = get_reader(reader, device=device)
  if summarize:
    summarizer_tokenizer, summarizer_model = get_summarizer(
        summarizer, device=device)
  timer = {"start": time.time()}
  dict_list: List[Dict] = []

  # Subset
  if summarize:
    reader_id = reader
    if '/' in reader:
      reader_id = reader.split('/')[1]
    subset = f"{db_name}.{collection_name}.{top_k}_{summarizer.split('/')[1]}.{ratio}_{reader_id}"
  else:
    reader_id = reader
    if '/' in reader:
      reader_id = reader.split('/')[1]
    subset = f"{db_name}.{collection_name}.{top_k}_{reader_id}"

  # Batch processing function
  def batch_qa(batch_data: Dict):
    batch_start = time.time()
    batch_zip = list(zip(batch_data['question'], batch_data['answer']))
    questions = [row[0] for row in batch_zip]
    answers = [row[1] for row in batch_zip]

    # Embedding
    start = time.time()
    if encoder == 'dpr':
      question_vectors = encode_dpr_question(
          encoder_tokenizer, encoder_model, questions)
      question_vectors = question_vectors.detach().cpu().numpy().tolist()
    print(f"({time.time() - start:.2f}s): encoding")

    # Retriever
    start = time.time()
    if summarize:
      limit = int(top_k) * ratio
    else:
      limit = top_k
    results = client.search(collection_name=collection_name,
                            data=question_vectors, limit=limit, output_fields=['title', 'text'])
    psgs_list: List[List[str]] = []
    for psgs in results:
      psgs_list.append([psg['entity']['text'] for psg in psgs])
    ctxs = ['\n'.join(psgs) for psgs in psgs_list]
    print(f"({time.time() - start:.2f}s): retrieval")

    # Summarizer
    summaries: List[str] = []
    if summarize:
      start = time.time()
      if ratio == 1:
        # Memory bound to batch_size
        summaries.extend(summarize_text(
            summarizer_tokenizer, summarizer_model, ctxs, device=device))
      else:
        # Memory bound to ratio
        summary_ctxs: List[str] = []
        for i, ctx in enumerate(ctxs):
          random.seed(ctx)
          random.shuffle(psgs_list[i])
          chunk_size = len(psgs_list[i]) // ratio
          print(chunk_size)
          for j in range(chunk_size):
            summary_ctxs.append('\n'.join(psgs_list[i][j*ratio:(j+1)*ratio]))
          summary_ctxs.append('\n'.join(psgs_list[i][-ratio:]))
          chunk_summaries = summarize_text(
              summarizer_tokenizer, summarizer_model, summary_ctxs, device=device)
          summaries.append('\n'.join(chunk_summaries))
      print(f"({time.time() - start:.2f}s): summarizing")

    # Reader
    start = time.time()
    if 'gpt' in reader:
      predicts = ask_openai(
          reader, questions, summaries if summarize else ctxs)
    else:
      predicts = ask_reader(reader_tokenizer, reader_model,
                            questions, summaries if summarize else ctxs, device=device)
    print(f"({time.time() - start:.2f}s): reading")

    for i, question in enumerate(questions):
      dict_list.append({
          'question': question,
          'answer': answers[i],
          'retrieved': ctxs[i],
          'summary': summaries[i] if summarize else None,
          'predicted': predicts[i]['answer'],
          'score': predicts[i]['score'],
      })

    print(f"({time.time() - batch_start:.2f}s): [total]")
    print(f"({time.time() - timer['start']:.2f}s) {len(dict_list)}")
    print(f"{subset}\n")
    return batch_data

  # Batch processing
  batched = qa_dataset.map(batch_qa, batched=True, batch_size=batch_size)
  for _ in batched:
    continue

  # Upload to HuggingFace Hub
  if token is not None:
    Dataset.from_list(dict_list).push_to_hub(
        token=token, repo_id=f'{user}/{dataset}-validation',
        config_name=subset)

  return 'Done'


@torch.inference_mode()
def chat(top_k=10, milvus_port='19530', milvus_user='resrer', milvus_host=config['MILVUS_HOST'],
         milvus_pw=config['MILVUS_PW'], collection_name='dpr_nq', db_name="psgs_w100", summarize=False) -> str:
  connections.connect(
      host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_pw)
  client = MilvusClient(user=milvus_user, password=milvus_pw,
                        uri=f"http://{milvus_host}:{milvus_port}", db_name=db_name)

  # Load models
  encoder_tokenizer, encoder_model = get_dpr_encoder()
  summarizer_tokenizer, summarizer_model = get_summarizer()
  reader_tokenizer, reader_model = get_reader()

  # Conversation loop
  while True:
    query = input("\nQuestion: ")
    if query == "exit":
      break

    # Embedding
    question_vectors = encode_dpr_question(
        encoder_tokenizer, encoder_model, [query])
    query_vector = question_vectors.detach().cpu().numpy().tolist()[0]

    # Retriever
    results = client.search(collection_name=collection_name, data=[
        query_vector], limit=top_k, output_fields=['title', 'text'])
    texts = [result['entity']['text'] for result in results[0]]
    ctx = '\n'.join(texts)
    print(f"\nRetrieved: {ctx}")

    # Reader
    if summarize:
      summaries = summarize_text(
          summarizer_tokenizer, summarizer_model, [f"{ctx}"])
      ctx = summaries[0]
      print(f"\nSummary: {ctx}")
    answers = ask_reader(reader_tokenizer, reader_model, [query], [ctx])
    print(f"\nAnswer: {answers[0]['answer']}")

  return 'Done'


if __name__ == '__main__':
  fire.Fire()

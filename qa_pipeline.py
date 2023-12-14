import time
import random
from typing import Dict, List

import fire
import torch
from pymilvus import MilvusClient, connections
from dotenv import dotenv_values
from datasets import load_dataset, Dataset
from tei import TEIClient

from dpr.embedding import encode_dpr_question, get_dpr_encoder
from resrer.reader import ask_reader, get_reader, ask_openai, ask_dpr_reader
from resrer.summarizer import summarize_text, get_summarizer

config = dotenv_values(".env")


@torch.inference_mode()
def dataset(top_k: int = 10, milvus_port='19530', summarize=False, dataset='nq_open', device='cuda',
            encoder='dpr', split='validation', summarizer='seonglae/resrer-bart-base',
            reader="facebook/dpr-reader-single-nq-base", ratio: int = 1, stream: bool = False,
            tei_host="localhost", tei_port='8080', tei_protocol="http", special_token=False,
            milvus_user='root', milvus_host=config['MILVUS_HOST'], milvus_pw=config['MILVUS_PW'],
            collection_name='dpr_nq', db_name="psgs_w100", token=config['HF_TOKEN'],
            batch_size=30, user='seonglae') -> str:
  connections.connect(
      host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_pw)
  client = MilvusClient(user=milvus_user, password=milvus_pw,
                        uri=f"http://{milvus_host}:{milvus_port}", db_name=db_name)

  qa_dataset = load_dataset(dataset, split=split, streaming=stream)

  # Load models
  if encoder == 'dpr':
    encoder_tokenizer, encoder_model = get_dpr_encoder(device=device)
  elif encoder == 'tei':
    teiclient = TEIClient(host=tei_host, port=tei_port, protocol=tei_protocol)
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
          encoder_tokenizer, encoder_model, questions, device=device)
      question_vectors = question_vectors.detach().cpu().numpy().tolist()
    elif encoder == 'tei':
      question_vectors = teiclient.embed_batch_sync(questions)
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
      sep = '<sep>' if special_token else '\n'
      input_texts = [questions[i] + sep + '\n'.join(psgs)
                     for i, psgs in enumerate(psgs_list)]
      start = time.time()
      if ratio == 1:
        # Memory bound to batch_size
        summaries.extend(summarize_text(
            summarizer_tokenizer, summarizer_model, input_texts, device=device))
      else:
        # Memory bound to ratio
        # TODO: multi dpr read mapping & question prefix
        summary_ctxs: List[str] = []
        for i, ctx in enumerate(input_texts):
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
      if 'dpr' in reader and not summarize:
        predicts = ask_dpr_reader(reader_tokenizer, reader_model,
                              questions, psgs_list, device=device)
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
def chat(top_k=10, milvus_port='19530', milvus_user='root', milvus_host=config['MILVUS_HOST'],
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

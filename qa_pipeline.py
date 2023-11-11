import fire
import torch
from pymilvus import MilvusClient, connections

from dpr.embedding import encode_dpr_question
from resrer.reader import ask_hf_reader
from dotenv import dotenv_values

config = dotenv_values(".env")


@torch.no_grad()
def qa(top_k=10, milvus_port='19530', milvus_user='root', milvus_host=config['MILVUS_HOST'],
       milvus_pw=config['MILVUS_PW'], collection_name='dpr_nq', db_name="psgs_w100") -> str:
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
    print(ctx)

    # Reader
    response = ask_hf_reader(query, ctx)
    print(response['answer'])

  return 'Done'


if __name__ == '__main__':
  fire.Fire(qa)

import time
from typing import Dict

import fire
from pymilvus import MilvusClient, connections, db, Index, CollectionSchema, FieldSchema, DataType
from datasets import load_dataset, Dataset
from tei import TEIClient
from dotenv import dotenv_values

config = dotenv_values(".env")

def dataset(dataset_id="wiki_dpr", milvus_user='root', milvus_pw=config['MILVUS_PW'],
            prefix="", subset='psgs_w100.nq.exact', stream=True,
            milvus_host=config['MILVUS_HOST'], milvus_port='19530', dim=768,
            db_name="psgs_w100", collection_name='dpr_nq', tei=False,
            tei_host="localhost", tei_port='8080', tei_protocol="http",
            batch_size=5000, start_index=None, end_index=None):

  # Load DB
  connections.connect(
      host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_pw)

  if db_name not in db.list_database():
    db.create_database(db_name)
  client = MilvusClient(user=milvus_user, password=milvus_pw,
                        uri=f"http://{milvus_host}:{milvus_port}", db_name=db_name)
  if collection_name not in client.list_collections():
    title = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024)
    text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096)
    vec = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim)
    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR,
                           is_primary=True, max_length=8)
    schema = CollectionSchema(
        fields=[id_field, vec, title, text], enable_dynamic_field=True)
    index_params = {
        'index_type': 'HNSW', 'index_param': {'M': 32, 'efConstruction': 512}, 'ef': 8192}
    client.create_collection_with_schema(collection_name=collection_name, schema=schema, index_params=index_params)
    collection_info = client.describe_collection(collection_name=collection_name)
    client._create_index(collection_name=collection_name, index_params=index_params, vec_field_name='vec')
    print(collection_info)

  # Load dataset
  dataset = load_dataset(dataset_id, subset, streaming=stream)['train']
  if not stream and end_index is not None:
    dataset = dataset[:int(end_index)]
    dataset = Dataset.from_dict(dataset)
  if not stream and start_index is not None:
    dataset = dataset[int(start_index):]
    dataset = Dataset.from_dict(dataset)

  # Batch processing function
  if tei:
    teiclient = TEIClient(host=tei_host, port=tei_port, protocol=tei_protocol)

  def batch_encode(batch_data: Dict):
    start = time.time()
    batch_zip = zip(batch_data['id'], batch_data['title'], batch_data['text'])
    print(batch_data.keys())
    rows = [{'id': row[0], 'title': row[1], 'text': row[2]}
            for row in batch_zip]
    input_texts = [f"{prefix}{row['title']}\n{row['text']}" for row in rows]
    if batch_data['embeddings'] is not None:
      embeddings = batch_data['embeddings']
    else:
      embeddings = teiclient.embed_batch_sync(input_texts)
    for i, row in enumerate(rows):
      row['vec'] = embeddings[i]
    client.insert(collection_name=collection_name, data=rows)
    print(
        f"Batched {len(batch_data['id'])}rows takes ({time.time() - start:.2f}s)")
    return

  # Batch processing
  batched = dataset.map(batch_encode, batched=True, batch_size=batch_size)
  for _ in iter(batched): continue


if __name__ == '__main__':
  fire.Fire()

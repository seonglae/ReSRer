import os
import re

import requests
import fire
import pyarrow.parquet as pq
from datasets import concatenate_datasets, Dataset
from dotenv import dotenv_values

from resrer.eval import evaluate_remote_dataset

config = dotenv_values(".env")


def evaluate(token=config['HF_TOKEN'], dataset='seonglae/nq_open-validation', match=None):
  headers = {"Authorization": f"Bearer {token}"}
  url = f"https://datasets-server.huggingface.co/splits?dataset={dataset}"
  response = requests.get(url, headers=headers, timeout=10)
  data = response.json()
  for split in data['splits']:
    if match:
      if not re.match(match, split['config']):
        continue
    result = evaluate_remote_dataset(dataset, split['config'])
    print(f"{split['config']}: {result}")
  return 'Done'


def upload(repo='seonglae/resrer-nq', folder='data/train/', token=config['HF_TOKEN']):
  parquets = os.listdir(folder)
  arrows = list(map(lambda path: pq.read_table(
      folder + path, memory_map=True), parquets))
  datasets = list(map(Dataset, arrows))
  dataset = concatenate_datasets(datasets)
  ds = dataset.filter(lambda row: row['summarization_text'] != '')
  print(ds)
  ds.push_to_hub(repo_id=repo, token=token)
  return 'Done'


if __name__ == '__main__':
  fire.Fire()

from typing import Dict, List
import time
import fire
from datasets import load_dataset, Dataset
import tiktoken
from dotenv import dotenv_values

from resrer.utils import split_token, Row

config = dotenv_values()


def split(dataset_id="wikipedia",  target='gpt-4', subset='20220301.en', stream=False,
          batch_size=5000, token=config['HF_TOKEN'], user='seonglae', split=256):
  encoder = tiktoken.encoding_for_model(target)

  # Load dataset
  dataset = load_dataset(dataset_id, subset, streaming=stream)['train']
  dict_list = []

  def batch_split(batch_data: Dict):
    start = time.time()
    batch_zip = zip(batch_data['id'], batch_data['title'],
                    batch_data['text'], batch_data['url'])
    print(batch_data.keys())
    rows: List[Row] = [{'id': row[0], 'title': row[1], 'text': row[2], 'url': row[3]}
                       for row in batch_zip]
    input_texts = [f"{row['title']}\n{row['text']}" for row in rows]
    dict_list.extend(split_token(encoder, rows, input_texts, split=split))
    print(
        f"{len(dict_list)}rows ({time.time() - start:.2f}s)")
    return {'query': input_texts}

  # Batch processing
  batched = dataset.map(batch_split, batched=True, batch_size=batch_size)
  for _ in batched:
    continue

  # Upload to HuggingFace Hub
  if token is not None:
    Dataset.from_list(dict_list).push_to_hub(config_name=target,
                                             token=token, repo_id=f'{user}/{dataset_id}-{split}')
  return 'done'


def count(dataset_id="wiki_dpr", target='gpt-4', subset='psgs_w100.nq.no_index.no_embeddings', stream=False,
          batch_size=5000, token=config['HF_TOKEN'], user='seonglae'):
  encoder = tiktoken.encoding_for_model(target)

  # Load dataset
  dataset = load_dataset(dataset_id, subset, streaming=stream)['train']
  dict_list = []
  token_map = {
      "~128": 0,
      "128~256": 0,
      "256~512": 0,
      "512~1024": 0,
      "1024~2048": 0,
      "2048~4096":  0,
      "4096~8192": 0,
      "8192~16384": 0,
      "16384~32768": 0,
      "32768~65536": 0,
      "65536~128000": 0,  # GPT4 Max
      "128000~": 0,
  }
  char_map = {
      "~512": 0,
      "512~1024": 0,
      "1024~2048": 0,
      "2048~4096":  0,
      "4096~8192": 0,
      "8192~16384": 0,
      "16384~32768": 0,
      "32768~65536": 0,  # Milvus Max
      "65536~": 0,
  }

  def batch_count(batch_data: Dict):
    start = time.time()
    batch_zip = zip(batch_data['id'], batch_data['title'], batch_data['text'])
    print(batch_data.keys())
    rows = [{'id': row[0], 'title': row[1], 'text': row[2]}
            for row in batch_zip]
    input_texts = [f"{row['title']}\n{row['text']}" for row in rows]
    tokens = encoder.encode_batch(input_texts)
    for i, row in enumerate(rows):
      row['token_length'] = len(tokens[i])
      row['text_length'] = len(row['text'])

      # Token length
      if row["token_length"] <= 128:
        token_map["~128"] += 1
      elif row["token_length"] <= 256:
        token_map["128~256"] += 1
      elif row["token_length"] <= 512:
        token_map["256~512"] += 1
      elif row["token_length"] <= 1024:
        token_map["512~1024"] += 1
      elif row["token_length"] <= 2048:
        token_map["1024~2048"] += 1
      elif row["token_length"] <= 4096:
        token_map["2048~4096"] += 1
      elif row["token_length"] <= 8192:
        token_map["4096~8192"] += 1
      elif row["token_length"] <= 16384:
        token_map["8192~16384"] += 1
      elif row["token_length"] <= 32768:
        token_map["16384~32768"] += 1
      elif row["token_length"] <= 65536:
        token_map["32768~65536"] += 1
      elif row["token_length"] <= 128000:
        token_map["65536~128000"] += 1
      else:
        token_map["128000~"] += 1
      # Text length
      if row["text_length"] <= 512:
        char_map["~512"] += 1
      elif row["text_length"] <= 1024:
        char_map["512~1024"] += 1
      elif row["text_length"] <= 2048:
        char_map["1024~2048"] += 1
      elif row["text_length"] <= 4096:
        char_map["2048~4096"] += 1
      elif row["text_length"] <= 8192:
        char_map["4096~8192"] += 1
      elif row["text_length"] <= 16384:
        char_map["8192~16384"] += 1
      elif row["text_length"] <= 32768:
        char_map["16384~32768"] += 1
      elif row["text_length"] <= 65536:
        char_map["32768~65536"] += 1
      else:
        char_map["65536~"] += 1
    dict_list.extend(rows)
    print(
        f"Batched {len(batch_data['id'])}rows takes ({time.time() - start:.2f}s)")
    return {'query': input_texts}

  # Batch processing
  batched = dataset.map(batch_count, batched=True, batch_size=batch_size)
  for _ in batched:
    continue

  # Upload to HuggingFace Hub
  if token is not None:
    Dataset.from_list(dict_list).push_to_hub(config_name=target,
                                             token=token, repo_id=f'{user}/{dataset_id}_token')

  print("Token count", token_map)
  print("Text count", char_map)
  total = sum(token_map.values())
  token_percent = {k: f'{v * 100 / total:.2f}%' for k, v in token_map.items()}
  char_percent = {k: f'{v * 100 / total:.2f}%' for k, v in char_map.items()}
  print("Token percent", token_percent)
  print("Text percent", char_percent)
  return token_percent, char_percent


if __name__ == '__main__':
  fire.Fire()

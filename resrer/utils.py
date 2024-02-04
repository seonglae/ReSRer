from typing import TypedDict, List
import asyncio
import time

from tiktoken import Encoding
from openai import AsyncOpenAI
from openai import APITimeoutError, Timeout, RateLimitError, APIError, APIConnectionError
from dotenv import dotenv_values


class Row(TypedDict):
  id: str
  title: str
  url: str
  text: str


config = dotenv_values(".env")
client = AsyncOpenAI(
    api_key=config['OPENAI_API_KEY'], organization=config['OPENAI_ORG'])


# OpenAI readrer
async def ask_openai_single(model, system_prompt: str, user_prompt: str) -> str:
  while True:
    try:
      res = await client.chat.completions.create(messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ], model=model, stream=False, timeout=20.0)

    except APITimeoutError as e:
      retry_time = e.retry_after if hasattr(e, "retry_after") else 30
      print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
      print('retry')
      continue

    except RateLimitError as e:
      retry_time = e.retry_after if hasattr(e, "retry_after") else 30
      print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      continue

    except APIError as e:
      retry_time = e.retry_after if hasattr(e, "retry_after") else 30
      print(f"API error occurred. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      continue

    except OSError as e:
      retry_time = 50  # Adjust the retry time as needed
      print(
          f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
      )
      time.sleep(retry_time)
      continue
    return str(res.choices[0].message.content)


async def ask_openai_batch(model, system_prompt: str, user_prompts: List[str]) -> List[str]:
  answers = await asyncio.gather(*[ask_openai_single(model, system_prompt, u) for u in user_prompts])
  return answers


def ask_openai(model, system_prompt: str, user_prompts: List[str]) -> List[str]:
  return asyncio.run(ask_openai_batch(model, system_prompt, user_prompts))


def split_token(encoder: Encoding, rows: List[Row], input_texts: List[str], split: int = 512) -> List[Row]:
  dict_list: List[Row] = []
  filtered_texts = []
  for input_text in input_texts:
    words = input_text.split(' ')
    words = list(filter(lambda word: len(word) < 1000, words))
    filtered_texts.append(' '.join(words))

  # Batch documents
  for i, text_tokenes in enumerate(encoder.encode_batch(filtered_texts)):
    row = rows[i]
    passages_count = int((len(text_tokenes) - 1) / split)

    # Passages from start
    for i in range(passages_count):
      tokens = text_tokenes[i * split:(i + 1) * split]

      # Append tokens until meet whitespace
      for token in text_tokenes[(i + 1) * split:]:
        if not encoder.decode_single_token_bytes(token).startswith(b' '):
          tokens.append(token)
        else:
          break

      # Unshift tokens until meet whitespace
      if not encoder.decode_single_token_bytes(text_tokenes[i * split]).startswith(b' '):
        for token in reversed(text_tokenes[:i * split]):
          if not encoder.decode_single_token_bytes(token).startswith(b' '):
            tokens.insert(0, token)
          else:
            tokens.insert(0, token)
            break
      dict_list.append({'id': f"{row['id']}_{i}", 'title': row['title'], 'url': row['url'],
                        'text': encoder.decode(tokens)})

    # Passages from end
    tokens = text_tokenes[-split:]
    if not encoder.decode_single_token_bytes(text_tokenes[0]).startswith(b' '):

      # Unshift tokens until meet whitespace
      for token in reversed(text_tokenes[:-split]):
        if not encoder.decode_single_token_bytes(token).startswith(b' '):
          tokens.insert(0, token)
        else:
          tokens.insert(0, token)
          break
    dict_list.append({'id': f"{row['id']}_{passages_count}", 'title': row['title'], 'url': row['url'],
                      'text': encoder.decode(tokens)})
  return dict_list

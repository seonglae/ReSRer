from typing import TypedDict, List
from tiktoken import Encoding


class Row(TypedDict):
  id: str
  title: str
  url: str
  text: str


def split_token(encoder: Encoding, rows: List[Row], input_texts: List[str], split: int = 512) -> List[Row]:
  dict_list: List[Row] = []

  # Batch documents
  for i, text_tokenes in enumerate(encoder.encode_batch(input_texts)):
    row = rows[i]
    passages_count = int((len(text_tokenes) - 1) / split)

    # Passages from start
    for i in range(passages_count):
      tokens = text_tokenes[i * split:(i + 1) * split]
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

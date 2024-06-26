from typing import List, Dict

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
  last_hidden = last_hidden_states.masked_fill(
      ~attention_mask[..., None].bool(), 0.0)
  return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_hf(input_texts: List[str], model_id: str = 'thenlper/gte-small',
              prefix: str = '', device='cuda'):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModel.from_pretrained(model_id).to(device)
  input_texts = [prefix + input_text for input_text in input_texts]
  # Tokenize the input texts
  batch_dict = tokenizer(input_texts, padding=True,
                         truncation=True, return_tensors='pt').to(device)
  outputs = model(**batch_dict)
  embeddings = average_pool(outputs.last_hidden_state,
                            batch_dict['attention_mask'])
  # normalize embeddings
  embeddings = F.normalize(embeddings)
  return embeddings

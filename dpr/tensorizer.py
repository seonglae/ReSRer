
from typing import Optional

from transformers import BertTokenizer
import torch

T = torch.Tensor


class Tensorizer():
  """
  Component for all text to model input data conversions and related utility methods
  """

  # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
  def text_to_tensor(
      self,
      text: str,
      title: Optional[str] = None,
      add_special_tokens: bool = True,
      apply_max_len: bool = True,
  ):
    raise NotImplementedError

  def get_pair_separator_ids(self) -> T:
    raise NotImplementedError

  def get_pad_id(self) -> int:
    raise NotImplementedError

  def get_attn_mask(self, tokens_tensor: T):
    raise NotImplementedError

  def is_sub_word_id(self, token_id: int):
    raise NotImplementedError

  def to_string(self, token_ids, skip_special_tokens=True):
    raise NotImplementedError

  def set_pad_to_max(self, do_pad: bool):
    raise NotImplementedError

  def get_token_id(self, token: str) -> int:
    raise NotImplementedError


class BertTensorizer(Tensorizer):
  def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True):
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.pad_to_max = pad_to_max

  def text_to_tensor(
      self,
      text: str,
      title: Optional[str] = None,
      add_special_tokens: bool = True,
      apply_max_len: bool = True,
  ):
    text = text.strip()
    # tokenizer automatic padding is explicitly disabled since its inconsistent behavior

    if title:
      token_ids = self.tokenizer.encode(
          title,
          text_pair=text,
          add_special_tokens=add_special_tokens,
          max_length=self.max_length if apply_max_len else 10000,
          pad_to_max_length=False,
          truncation=True,
      )
    else:
      token_ids = self.tokenizer.encode(
          text,
          add_special_tokens=add_special_tokens,
          max_length=self.max_length if apply_max_len else 10000,
          pad_to_max_length=False,
          truncation=True,
      )

    seq_len = self.max_length
    if self.pad_to_max and len(token_ids) < seq_len:
      token_ids = token_ids + \
          [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
    if len(token_ids) >= seq_len:
      token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
      token_ids[-1] = self.tokenizer.sep_token_id

    return torch.tensor(token_ids)

  def get_pair_separator_ids(self) -> T:
    return torch.tensor([self.tokenizer.sep_token_id])

  def get_pad_id(self) -> int:
    return self.tokenizer.pad_token_id

  def get_attn_mask(self, tokens_tensor: T) -> T:
    return tokens_tensor != self.get_pad_id()

  def is_sub_word_id(self, token_id: int):
    token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
    return token.startswith("##") or token.startswith(" ##")

  def to_string(self, token_ids, skip_special_tokens=True):
    return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

  def set_pad_to_max(self, do_pad: bool):
    self.pad_to_max = do_pad

  def get_token_id(self, token: str) -> int:
    return self.tokenizer.vocab[token]

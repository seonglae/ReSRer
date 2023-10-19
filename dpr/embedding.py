from typing import List
import torch

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, logging

logging.set_verbosity_error()


def encode_dpr_question(question: str, model_id="facebook/dpr-question_encoder-single-nq-base") -> torch.FloatTensor:
  """Encode a question using DPR question encoder.
  https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DPRQuestionEncoder

  Args:
      question (str): question string to encode
      model_id (str, optional): Default for NQ or "facebook/dpr-question_encoder-multiset-base
  """
  tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_id)
  model = DPRQuestionEncoder.from_pretrained(model_id)
  input_ids = tokenizer(question, return_tensors="pt")["input_ids"]
  embeddings: torch.FloatTensor = model(input_ids).pooler_output
  return embeddings

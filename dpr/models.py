from typing import List
import torch

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizerFast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer


def get_best_spans(start_logits: List, end_logits: List) -> List:
  """
  Finds the best answer span for the extractive Q&A model
  """
  scores = []
  for (i, s) in enumerate(start_logits):
    for (j, e) in enumerate(end_logits[i:]):
      scores.append(((i, i + j), s + e))

  scores = sorted(scores, key=lambda x: x[1], reverse=True)
  return scores


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


def read_dpr(questions: List[str], titles: List[str], texts: List[str], model_id="facebook/dpr-reader-single-nq-base"):
  """Read and find answer using DPR reader.

  Args:
      questions (List[str]):
      titles (List[str]):
      texts (List[str]):
      model_id (str, optional): Defaults for NQ or "facebook/dpr-reader-multiset-base".
  """
  tokenizer = DPRReaderTokenizerFast.from_pretrained(model_id)
  model = DPRReader.from_pretrained(model_id)
  encoded_inputs = tokenizer(
      questions=questions,
      titles=titles,
      texts=texts,
      return_tensors="pt",
  )
  outputs = model(**encoded_inputs)
  start_logits = outputs.start_logits.detach().numpy().tolist()[0]
  end_logits = outputs.end_logits.detach().numpy().tolist()[0]
  relevance_logits = outputs.relevance_logits.detach().numpy().tolist()[0]
  return (start_logits, end_logits, relevance_logits)


def encode_dpr_ctx(ctx: str, model_id="facebook/dpr-ctx_encoder-single-nq-base") -> torch.FloatTensor:
  """Encode a context using DPR context encoder.
  https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DPRContextEncoder

  Args:
      ctx (str): context string to encode
      model_id (str, optional): Default for NQ or "facebook/dpr-ctx_encoder-multiset-base"
  """
  tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_id)
  model = DPRContextEncoder.from_pretrained(model_id)
  input_ids = tokenizer(ctx, return_tensors="pt")["input_ids"]
  embeddings: torch.FloatTensor = model(input_ids).pooler_output
  return embeddings

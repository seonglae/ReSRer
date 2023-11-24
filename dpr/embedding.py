from typing import List, Tuple
import torch

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, logging

logging.set_verbosity_error()


def encode_dpr_question(tokenizer: DPRQuestionEncoderTokenizer, model: DPRQuestionEncoder, questions: List[str], device="cuda") -> torch.FloatTensor:
  """Encode a question using DPR question encoder.
  https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DPRQuestionEncoder

  Args:
      question (str): question string to encode
      model_id (str, optional): Default for NQ or "facebook/dpr-question_encoder-multiset-base
  """
  batch_dict = tokenizer(questions, return_tensors="pt",
                         padding=True, truncation=True,).to(device)
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    embeddings: torch.FloatTensor = model(**batch_dict).pooler_output
  return embeddings


def get_dpr_encoder(model_id="facebook/dpr-question_encoder-single-nq-base", device="cuda") -> Tuple[DPRQuestionEncoder, DPRQuestionEncoderTokenizer]:
  """Encode a question using DPR question encoder.
  https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DPRQuestionEncoder

  Args:
      question (str): question string to encode
      model_id (str, optional): Default for NQ or "facebook/dpr-question_encoder-multiset-base
  """
  tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_id)
  model = DPRQuestionEncoder.from_pretrained(model_id).to(device)
  return tokenizer, model

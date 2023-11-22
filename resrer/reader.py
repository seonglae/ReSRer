from typing import TypedDict, List, Dict
from re import sub

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DPRReaderTokenizer, DPRReader, logging
from transformers import QuestionAnsweringPipeline

max_answer_len = 8
logging.set_verbosity_error()


class AnswerInfo(TypedDict):
  score: float
  start: int
  end: int
  answer: str


@torch.inference_mode()
def ask_reader(tokenizer: AutoTokenizer, model: AutoModelForQuestionAnswering,
               questions: List[str], ctxs: List[str]) -> List[AnswerInfo]:
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    pipeline = QuestionAnsweringPipeline(
        model=model, tokenizer=tokenizer, device='cuda', max_answer_len=max_answer_len)
    answer_infos: List[AnswerInfo] = pipeline(
        question=questions, context=ctxs)
  for answer_info in answer_infos:
    answer_info['answer'] = sub(r'[.\(\)"\',]', '', answer_info['answer'])
  return answer_infos


def get_reader(model_id="mrm8488/longformer-base-4096-finetuned-squadv2"):
  tokenizer = DPRReaderTokenizer.from_pretrained(model_id)
  model = DPRReader.from_pretrained(model_id).to(0)
  return tokenizer, model

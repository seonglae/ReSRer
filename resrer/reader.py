from typing import TypedDict, List, Dict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import QuestionAnsweringPipeline

max_answer_len = 5


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
  return answer_infos


def get_reader(model_id="mrm8488/longformer-base-4096-finetuned-squadv2"):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForQuestionAnswering.from_pretrained(model_id).to(0)
  return tokenizer, model

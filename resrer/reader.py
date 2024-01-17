from typing import TypedDict, List, Optional
from re import sub

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DPRReaderTokenizer, DPRReader, logging
from transformers import QuestionAnsweringPipeline

from dpr.reader import get_best_spans
from dpr.tensorizer import BertTensorizer
from resrer.utils import ask_openai

max_answer_len = 8
logging.set_verbosity_error()
class AnswerInfo(TypedDict):
  score: float
  start: Optional[int]
  end: Optional[int]
  answer: str


@torch.inference_mode()
def ask_dpr_reader(tokenizer: AutoTokenizer, model: AutoModelForQuestionAnswering,
                   questions: List[str], psgs: List[List[str]], device='cuda') -> List[AnswerInfo]:
  top_k = len(psgs[0])
  answer_candidates = []
  for i in range(top_k):
    psgs_list = [psg[i] for psg in psgs]
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
      pipeline = QuestionAnsweringPipeline(
          model=model, tokenizer=tokenizer, device=device, max_answer_len=max_answer_len)
      answer_infos: List[AnswerInfo] = pipeline(
          question=questions, context=psgs_list)
    if not isinstance(answer_infos, list):
      answer_infos = [answer_infos]

    # Remove special tokens for DPR reader
    for answer_info in answer_infos:
      answer_info['answer'] = sub(r'[.\(\)"\',]', '', answer_info['answer'])
    answer_candidates.append(answer_infos)

  # Select best answer
  answer_infos = [max((answer_candidates[k][i]
                      for k in range(top_k)), key=lambda a: a['score']) for i in range(len(psgs))]
  # tensorizer = BertTensorizer(tokenizer, 320)
  # for k in range(top_k):
  #   get_best_spans(tensorizer, [answer_candidates[k][i]['start'] for i in range(len(psgs))],
  #                  [answer_candidates[k][i]['end'] for i in range(len(psgs))], tensorizer.text_to_tensor(
  #       questions[0]), max_answer_len, 0, 0, top_spans=10)
  print(answer_infos[0])
  return answer_infos


@torch.inference_mode()
def ask_reader(tokenizer: AutoTokenizer, model: AutoModelForQuestionAnswering,
               questions: List[str], ctxs: List[str], device='cuda') -> List[AnswerInfo]:
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    pipeline = QuestionAnsweringPipeline(
        model=model, tokenizer=tokenizer, device=device, max_answer_len=max_answer_len)
    answer_infos: List[AnswerInfo] = pipeline(
        question=questions, context=ctxs)
  if not isinstance(answer_infos, list):
    answer_infos = [answer_infos]
  for answer_info in answer_infos:
    answer_info['answer'] = sub(r'[.\(\)"\',]', '', answer_info['answer'])
  return answer_infos


def get_reader(model_id="facebook/dpr-reader-single-nq-base", device="cuda"):
  tokenizer = DPRReaderTokenizer.from_pretrained(model_id)
  model = DPRReader.from_pretrained(model_id).to(device)
  return tokenizer, model


def ask_openai_reader(model, questions: List[str], ctxs: List[str]) -> List[AnswerInfo]:
  system_prompt = '''###Instruction###
Extract a concise noun-based answer from the provided context for the question. Your answer should be under three words and extracted directly from a context of no more than five words. You can analyze the context step by step to derive the answer. Avoid using prefixes that indicate the type of answer; simply present the shortest relevant answer span from the context.
'''
  user_prompts = [f'###Question###\n{q}\n\n###Context###\n{ctxs[i]}' for i, q in enumerate(questions)]
  answers = ask_openai(model, system_prompt, user_prompts)
  return [{'answer': a, 'score': 0, 'start': 0, 'end': 0} for a in answers]

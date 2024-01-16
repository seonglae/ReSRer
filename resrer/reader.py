from typing import TypedDict, List, Optional
from re import sub
import asyncio

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DPRReaderTokenizer, DPRReader, logging
from transformers import QuestionAnsweringPipeline
from openai import AsyncOpenAI, APITimeoutError
from dotenv import dotenv_values

from dpr.reader import get_best_spans
from dpr.tensorizer import BertTensorizer

max_answer_len = 8
logging.set_verbosity_error()
config = dotenv_values(".env")

client = AsyncOpenAI(api_key=config['OPENAI_API_KEY'],)


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


# OpenAI readrer
async def ask_openai_single(model, question: str, ctx: str) -> AnswerInfo:
  system = 'question Instructions: Extract noun answer for question from context under 3 words. You must extract answer from a context at most 5 words.'
  user = f'question: {question}\ncontext: {ctx}'
  while True:
    try:
      res = await client.chat.completions.create(messages=[
          {"role": "system", "content": system},
          {"role": "user", "content": user}
      ], model=model, stream=False, timeout=5.0)
    except APITimeoutError as _:
      print('retry')
      continue
    return {"answer": str(res.choices[0].message.content), "score": 0, "start": 0, "end": 0}


async def ask_openai_batch(model, questions: List[str], ctxs: List[str]) -> List[AnswerInfo]:
  answers = await asyncio.gather(*[ask_openai_single(model, questions[i], ctx) for i, ctx in enumerate(ctxs)])
  return answers


def ask_openai(model, questions: List[str], ctxs: List[str]) -> List[AnswerInfo]:
  return asyncio.run(ask_openai_batch(model, questions, ctxs))

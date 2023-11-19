from typing import TypedDict, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class AnswerInfo(TypedDict):
  score: float
  start: int
  end: int
  answer: str


def get_answers(tokenizer: AutoTokenizer, output: QuestionAnsweringModelOutput, batch_dict: Dict) -> AnswerInfo:
  start = int(torch.argmax(output.start_logits))
  end = int(torch.argmax(output.end_logits)) + 1
  score = float(torch.max(output.start_logits) + torch.max(output.end_logits))
  answer = tokenizer.decode(tokenizer.encode(
      tokenizer.decode(batch_dict['input_ids'][0][start:end])))
  return AnswerInfo(score=score, start=start, end=end, answer=answer)


@torch.no_grad()
def ask_reader(tokenizer: AutoTokenizer, model: AutoModelForQuestionAnswering,
               questions: List[str], ctxs: List[str]) -> List[AnswerInfo]:
  inputs = tokenizer(questions, ctxs, return_tensors='pt', stride=50,
                     return_offsets_mapping=True, return_overflowing_tokens=True,
                     truncation='only_second', padding=True).to('cuda')
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    res = model(inputs['input_ids'])
  print(res)
  return res


def get_reader(model_id="mrm8488/longformer-base-4096-finetuned-squadv2"):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForQuestionAnswering.from_pretrained(model_id).to('cuda')
  return tokenizer, model

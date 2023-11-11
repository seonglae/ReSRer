from typing import TypedDict
from transformers import pipeline


class AnswerInfo(TypedDict):
  score: float
  start: int
  end: int
  answer: str


def ask_hf_reader(question: str, ctx: str, model_id: str = "mrm8488/longformer-base-4096-finetuned-squadv2") -> AnswerInfo:
  nlp = pipeline('question-answering', model=model_id, tokenizer=model_id)
  QA_input = {
      'question': question,
      'context': ctx
  }
  res = nlp(QA_input)
  return res

# GPT4 reader

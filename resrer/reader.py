from typing import TypedDict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class AnswerInfo(TypedDict):
  score: float
  start: int
  end: int
  answer: str


def ask_hf_reader(question: str, ctx: str, model_id: str = "deepset/roberta-base-squad2") -> AnswerInfo:
  nlp = pipeline('question-answering', model=model_id, tokenizer=model_id)
  QA_input = {
      'question': question,
      'context': ctx
  }
  res = nlp(QA_input)
  return res

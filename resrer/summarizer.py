from typing import List, Tuple, Union
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import PegasusXForConditionalGeneration, PegasusTokenizerFast
import torch

from resrer.utils import ask_openai


def summarize_text(tokenizer: Union[PegasusTokenizerFast, BartTokenizerFast],
                   model: Union[PegasusXForConditionalGeneration, BartForConditionalGeneration],
                   psgs_list: List[List[str]], summarizer: str, questions: List[str], device="cuda",
                   special_token=False) -> List[str]:
  input_texts = ['\n'.join(psgs) for psgs in psgs_list]
  if 'gpt' in summarizer:
    return gpt_summarize_text(summarizer, input_texts, questions)

  sep = '<sep>' if special_token else '\n'
  input_texts = [questions[i] + sep +
                 text for i, text in enumerate(input_texts)]
  inputs = tokenizer(input_texts, padding=True,
                     return_tensors='pt', truncation=True).to(device)
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
    summary_ids = model.generate(inputs["input_ids"])
  summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False, batch_size=len(input_texts))
  return summaries


def gpt_summarize_text(model: str, input_texts: List[str], questions: List[str]) -> List[str]:
  system_prompt = '''###Instruction###
Rewrite the given passages to be easier for the reader answering the given question.
The rewrited text should be half the total length of the original passages. Your response must be at least 200 words long.
The given passages are related about the question topic.
Use only information in the document.
Reduce the noise unrelated to answer the question.
Remove unrelated phrases and sentences to answer the question.
Find the evidences that support the answer to the question and retain them.
Print only the rewrited texts
The final answer for this question is contained is the passages so maintain the exact span of answer smaller than 5 words.
'''
  user_prompts = [
      f'###Question###{questions[i]}\n\n###Passages###\n{t}' for i, t in enumerate(input_texts)]
  return ask_openai(model, system_prompt, user_prompts)


def get_summarizer(model_id="seonglae/resrer-bart-base", device="cuda") -> Tuple[Union[PegasusTokenizerFast,
                                                                                       BartTokenizerFast],
                                                                                 Union[PegasusXForConditionalGeneration,
                                                                                       BartForConditionalGeneration]]:
  if 'gpt' in model_id:
    return None, None
  elif 'bart' in model_id:
    tokenizer = BartTokenizerFast.from_pretrained(model_id)
    model = BartForConditionalGeneration.from_pretrained(
        model_id, min_length=256, max_length=512).to(device)
  elif 'pegasus' in model_id:
    tokenizer = PegasusTokenizerFast.from_pretrained(model_id)
    model = PegasusXForConditionalGeneration.from_pretrained(
        model_id, min_length=256, max_length=512).to(device)
  model = torch.compile(model)
  return tokenizer, model

# OpenAI summarizer

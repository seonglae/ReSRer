from typing import List
from transformers import AutoTokenizer, PegasusXForConditionalGeneration, PegasusTokenizer
import torch


def summarize_text(model: PegasusXForConditionalGeneration, tokenizer: PegasusTokenizer,
                   input_texts: List[str]):
  inputs = tokenizer(input_texts, max_length=1024, padding=True,
                     return_tensors='pt', truncation=True).to('cuda')
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    summary_ids = model.generate(inputs["input_ids"])
  summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False, batch_size=len(input_texts))
  return summaries


def get_summarizer(model_id="pszemraj/pegasus-x-large-book-summary"):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = PegasusXForConditionalGeneration.from_pretrained(model_id).to('cuda')
  model = torch.compile(model)
  return tokenizer, model


# GPT4 reader

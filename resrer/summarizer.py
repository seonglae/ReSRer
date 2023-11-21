from typing import List, Tuple
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizerFast
import torch


def summarize_text(tokenizer: BartTokenizerFast, model: BartForConditionalGeneration,
                   input_texts: List[str]):
  inputs = tokenizer(input_texts, padding=True,
                     return_tensors='pt', truncation=True).to(1)
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    summary_ids = model.generate(inputs["input_ids"])
  summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False, batch_size=len(input_texts))
  return summaries


def get_summarizer(model_id="ccdv/lsg-bart-base-4096-multinews") -> Tuple[BartTokenizerFast, BartForConditionalGeneration]:
  tokenizer = BartTokenizerFast.from_pretrained(model_id)
  model = BartForConditionalGeneration.from_pretrained(model_id).to(1)
  model = torch.compile(model)
  return tokenizer, model


# OpenAI reader

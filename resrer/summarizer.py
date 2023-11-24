from typing import List, Tuple, Union
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import PegasusXForConditionalGeneration, PegasusTokenizerFast
import torch


def summarize_text(tokenizer: Union[PegasusTokenizerFast, BartTokenizerFast],
                   model: Union[PegasusXForConditionalGeneration, BartForConditionalGeneration],
                   input_texts: List[str], device="cuda"):
  inputs = tokenizer(input_texts, padding=True,
                     return_tensors='pt', truncation=True).to(device)
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    summary_ids = model.generate(inputs["input_ids"])
  summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False, batch_size=len(input_texts))
  return summaries


def get_summarizer(model_id="seonglae/resrer-bart-base", device="cuda") -> Tuple[Union[PegasusTokenizerFast,
                                                                        BartTokenizerFast],
                                                                  Union[PegasusXForConditionalGeneration,
                                                                        BartForConditionalGeneration]]:
  if 'bart' in model_id:
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

from transformers import pipeline


def summarize_text(input: str, model_id: str = "pszemraj/pegasus-x-large-book-summary") -> str:
  nlp = pipeline('summarization', model=model_id,
                 tokenizer=model_id, device='cuda')
  res = nlp(input)[0]['summary_text']
  return res

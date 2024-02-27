from typing import List, Tuple, Union
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import PegasusXForConditionalGeneration, PegasusTokenizerFast
import torch

from resrer.utils import ask_openai

instructions = {
    "v2": '''###Instruction###
Rewrite the given passages to be easier for the reader answering the given question.
The rewrited text should be half the total length of the original passages. Your response must be at least 200 words long.
The given passages are related about the question topic.
Use only information in the document.
Reduce the noise unrelated to answer the question.
Remove unrelated phrases and sentences to answer the question.
Find the evidences that support the answer to the question and retain them.
Print only the rewrited texts
The final answer for this question is contained is the passages so maintain the exact span of answer smaller than 5 words.
''',
    "v6": '''###Instruction###
Condense the provided passages to focus on key elements directly answering the question. Your summary should be a third of the original passages' length and at least 100 words. Highlight critical information and evidence supporting the answer. Avoid generalizations or unrelated details. Ensure the final answer is present in the summary, keeping the exact span of the answer to under five words. Present the summary in a clear, bullet-point format for each key element related to the question. Aim for a balance between conciseness and completeness.
''',
    'v7': '''###Instruction###
Condense the provided passages to focus on key elements directly answering the question. Your summary should be a third of the original passages' length and at least 150 words. Highlight critical information and evidence supporting the answer. Avoid generalizations or unrelated details. Ensure the final answer is present in the summary, keeping the exact span of the answer to under five words. Present the summary in a clear, bullet-point format for each key element related to the question. Aim for a balance between conciseness and completeness.
''',
    'v8': '''###Instruction###
Your task is to rewrite the provided passages to enhance their specificity and precision in relation to the question. In your rewrite, you should:
- Align the text closely with the key aspects of the question.
- Your summary should be a third of the original passages' length and at least 200 words..
- Prioritize information most likely to contain or support the answer.
- Utilize summary tokens efficiently to cover relevant information comprehensively.
- Ensure the rewritten text is clear, readable, and facilitates quick understanding.
- Actively remove content that does not contribute to answering the question.
Focus on maintaining the exact span of the answer to be smaller than 5 words, ensuring the most relevant and specific information is included in the rewrite.
''',
    'v9': '''###Instruction###
Your task is to rewrite the provided passages to enhance their specificity and precision in relation to the question. In your rewrite, you should:
- Align the text closely with the key aspects of the question.
- Your summary should be a half of the original passages' length and at least 150 words.
- Prioritize information most likely to contain or support the answer.
- Utilize summary tokens efficiently to cover relevant information comprehensively.
- Ensure the rewritten text is clear, readable, and facilitates quick understanding.
- Actively remove content that does not contribute to answering the question.
- Understand the intent of the question and summarize to make it easier to find an answer according to the intent.
Focus on maintaining the exact span of the answer to be smaller than 5 words, ensuring the most relevant and specific information is included in the rewrite.
''', 'v10': '''###Instruction###
Rewrite the provided passages to enhance their precision and clarity in addressing the question. The rewritten text should be approximately half the length of the original passages. Your response must be at least 200 words long. Ensure that the revised text is directly related to the topic of the question. Utilize only the information provided in the document. Remove any irrelevant phrases and sentences. Identify and retain evidence that supports the answer. Print only the rewritten text. Make sure the exact span of the answer is smaller than 5 words.''',
'v11': '''###Instruction###
To improve the precision and clarity of the given passages in relation to the question, rewrite them as follows:
- Align the text closely with the key aspects of the question.
- Your summary should be half the length of the original passages and contain at least 150 words.
- Prioritize information that is most likely to contain or support the answer.
- Use summary tokens efficiently to cover relevant information comprehensively.
- Ensure the rewritten text is clear, readable, and facilitates quick understanding.
- Remove any content that does not contribute to answering the question.
- Summarize the passages according to the intent of the question to make it easier to find an answer.
- Focus on maintaining the exact span of the answer under 5 words, while including the most relevant and specific information in the rewrite.
By following these guidelines, you will create a more structured and precise version of the passages, helping another language model effectively solve open-domain, single-hop, short-answer question answering problems.''',
'v12': '''###Instruction###In order to enhance the precision and specificity of the given passages in relation to the question, your task is to rewrite them with the following guidelines:
- Align the text closely with the key aspects of the question.
- Your summary should be half the length of the original passages and contain at least 150 words.
- Prioritize information that is most likely to contain or support the answer.
- Use summary tokens efficiently to cover relevant information comprehensively.
- Ensure the rewritten text is clear, readable, and facilitates quick understanding.
- Remove any content that does not contribute to answering the question.
- Summarize the passages according to the intent of the question to make it easier to find an answer.
- Focus on maintaining the exact span of the answer under 5 words, while including the most relevant and specific information in the rewrite.
By following these guidelines, you will create a more structured and precise version of the passages, aiding another language model in performing open-domain, single-hop, short-answer question answering problems more effectively and accurately.''',
'v13':'Follow these instructions to accurately summarize or rewrite the randomly extracted and imperfect passages from Wikipedia. The objective is to create concise and structured text that will assist another language model in effectively solving open-domain, single-hop, short-answer question answering problems. Pay attention to enhancing precision, clarity, and structure to provide accurate and relevant information in the summaries or rewrites.',
'v14':'''###Instruction###
Your task is to rewrite the provided passages to enhance their specificity and precision in relation to the question. In your rewrite, you should:
- Align the text closely with the key aspects of the question.
- Your summary should be a third of the original passages' length.
- Remove spans that might confuse the reader with the correct answer.
- Prioritize information most likely to contain or support the answer.
- Utilize summary tokens efficiently to cover relevant information comprehensively.
- Ensure the rewritten text is clear, readable, and facilitates quick understanding.
- Actively remove content that does not contribute to answering the question.
Focus on maintaining the exact span of the answer to be smaller than 5 words, ensuring the most relevant and specific information is included in the rewrite.
''',
}


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
  version = model.split('.')[-1]
  if version in instructions:
    model = '.'.join(model.split('.')[:-1])

  system_prompt = instructions[version]
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

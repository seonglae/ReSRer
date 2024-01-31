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
    "v7": '''###Instruction###
Refine the provided passages with a focus on the key elements that directly answer the question. Your summary should aim to be about one-third the length of the original passages, but not less than 100 words. Adhere to these guidelines:
1. **Directly Address the Question**: Extract and emphasize information that directly answers the question. The final answer, under five words, must be identifiable in your summary.
2. **Structured Bullet-Point Format**: Present key information in a structured, bullet-point format. Each bullet point should correspond to a specific element or piece of evidence related to the question. This will aid in clarity and ease of understanding.
3. **Preserve Critical Details**: While summarizing, ensure that crucial information and terms, especially those that contribute to the EM score, are retained without alteration.
4. **Eliminate Redundant or Irrelevant Information**: Remove any content that does not contribute to answering the question, thus reducing the length of the text and focusing on relevant details.
5. **Coherent and Concise Summary**: The summary should be coherent, linking bullet points logically. Aim for a balance between brevity and comprehensive coverage of the necessary details.
6. **Continuous Improvement Based on Performance Data**: Regularly analyze the performance of this summarization approach, particularly the EM rates, and refine it accordingly to enhance effectiveness.
The goal is to provide a clear, concise, and relevant summary that maximizes EM retention, aiding the reader in quickly identifying the precise answer.''',
    "v8": '''###Instruction###Focus on condensing the provided passages to highlight the key elements that directly answer the question. Keep these guidelines in mind while summarizing:
1. **Concentrate on Exact Match (EM) Retention**: Prioritize the preservation of exact phrases and terms from the original text that are crucial for the question. This is essential for maximizing the EM rate.
2. **Structured and Concise Summary**: Aim for a summary that is about a third of the original passages' length, with a minimum of 100 words. Present the information in a clear, bullet-point format, focusing on different key elements or evidence related to the question.
3. **Eliminate Extraneous Information**: Carefully remove content that doesn't contribute to answering the question. This will help in reducing the length of the text while maintaining focus on relevant details.
4. **Ensure Coherence and Clarity**: While being concise, the summary should be coherent and easily understandable. Arrange the bullet points in a logical sequence that guides the reader to the answer.
5. **Continuous Improvement and Adjustment**: Regularly review and adjust your summarization technique based on its performance, particularly the EM and F1 scores, to enhance its effectiveness.
Remember, the objective is to provide a summary that is both concise and rich in crucial details, aiding in the quick identification of the precise answer.''',
'v9':'''###Instruction###
Create a summary of the provided passages with a sharp focus on the most relevant information that directly answers the question. Follow these revised guidelines to enhance precision and clarity:
1. **Zero in on Direct Answers**: Scrutinize the passages to identify the exact phrases and terms that directly respond to the question. Make sure these elements are central in your summary.
2. **Depth in Relevant Summarization**: Give a detailed summary for the passages that contain direct answers to the question. Summarize less relevant passages more briefly. Aim for the summary to be one-third the length of the original text, but ensure it is at least 100 words.
3. **Enhanced Structured Format**: Use a clear, structured format such as bullet points, ensuring each point succinctly encapsulates a direct piece of evidence or a key term related to the question.
4. **Rigorous Exclusion of Irrelevancies**: Actively remove any part of the text that doesn’t contribute to directly answering the question. This includes redundant phrases or loosely related information.
5. **Strengthen Logical Progression**: Organize the summary so that it logically leads the reader to the answer, with a clear and coherent structure that connects the bullet points or paragraphs.
6. **Direct and Unambiguous Language**: Employ straightforward and unambiguous language. Avoid any form of abstraction or generalization. Stick to the specific details and terms that are necessary for the answer.
7. **Review for Precision and Completeness**: After summarizing, review the content to ensure that it precisely captures the essential information from the original text and is complete in its coverage of these details.
The objective is to produce a summary that is not only precise in capturing the exact answer but is also comprehensive and clear, effectively guiding the reader to understand and identify the specific answer.
''',
'v10': '''###Instruction###
Efficiently summarize the provided passages, ensuring to extract and emphasize the elements directly relevant to answering the question. Adhere to these guidelines for a refined and targeted summary:
1. **Direct Extraction of Key Information**: Meticulously scan the passages to isolate the exact phrases and specific terms that answer the question directly. These elements are critical and should be prominently included in your summary.
2. **In-Depth Focus on Relevant Passages**: Offer a detailed summary for sections that contain direct answers to the question. For other sections, provide a concise overview. Aim for the summary to be approximately one-third of the original text's length, with a minimum of 100 words.
3. **Enhanced Structured Format for Clarity**: Present the summary in a structured format, such as bullet points or short paragraphs, where each point or paragraph concisely conveys a key piece of evidence or a crucial term related to the question.
4. **Rigorous Elimination of Irrelevant Material**: Actively remove any content that does not directly contribute to answering the question, including redundant or off-topic information.
5. **Logical Sequencing for Coherence**: Ensure the summary is logically organized, with a clear progression of ideas that seamlessly guides the reader to the answer.
6. **Utilization of Clear and Specific Language**: Use clear, specific, and unambiguous language in the summary. Avoid abstractions and maintain focus on the concrete details necessary for the answer.
7. **Critical Review for Precision and Completeness**: After summarizing, critically review the content to confirm that it precisely captures the essential information and fully addresses the details related to the question.
The objective is to craft a summary that accurately reflects the most crucial elements of the original text, particularly those directly responding to the question, in a coherent and accessible format.
''',
'v11': '''###Instruction###
Systematically summarize the passages to extract the most relevant information directly answering the question. Apply these focused guidelines for a more targeted and effective summary:
1. **Isolate and Emphasize Critical Information**: Diligently identify the exact phrases and terms from the passages that directly answer the question. These specific elements should form the core of your summary.
2. **Concentrated Summarization of Pertinent Sections**: Deliver a thorough summary for portions of the text directly responding to the question. Summarize the other sections more generally. The summary should be about one-third the length of the original text, but ensure a minimum of 100 words.
3. **Structured and Clear Presentation**: Utilize a structured format, like bullet points or concise paragraphs, to clearly present each key point or evidence that is relevant to the question.
4. **Eliminate All Non-Essential Content**: Aggressively cut out any part of the text that doesn't directly contribute to the answer. Avoid including any irrelevant or superfluous information.
5. **Maintain Logical Order and Connectivity**: Structure the summary in a logical manner, ensuring a coherent flow from one point to the next, leading the reader towards the answer.
6. **Precision and Directness in Language**: Choose language that is precise, clear, and direct. Steer clear of vague expressions and focus on specific, concrete details that are essential for the answer.
7. **Review for Accuracy and Thoroughness**: After summarizing, thoroughly review the content to verify its accuracy in capturing essential information and its completeness in covering the details pertinent to the question.
The aim is to produce a summary that not only precisely captures and highlights the key information from the original text but also presents it in a way that is straightforward and easily understandable, directly aiding in identifying the specific answer.'''
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

import re
import string
from typing import List

import unicodedata
import tiktoken
import numpy as np

from evaluate import evaluator, QuestionAnsweringEvaluator
from datasets import load_dataset, Dataset


encoder = tiktoken.encoding_for_model('gpt-4')


def exact_match(answers, prediction):
  """ Facebook DPR exact match metric.
  https://github.com/facebookresearch/DPR/blob/main/train_extractive_reader.py#L253
  Args:
      answers (_type_): _description_
      prediction (_type_): _description_

  Returns:
      _type_: _description_
  """
  if any((normalize(answer) == normalize(prediction) for answer in answers)):
    return 1
  if ',' in prediction:
    return np.mean([float(any((normalize(answer) == normalize(p)) for answer in answers)) for p in prediction.split(',')])
  if 'and' in prediction:
    return np.mean([float(any((normalize(answer) == normalize(p)) for answer in answers)) for p in prediction.split('and')])
  return 0

def exact_contain(answers, context):
  return any((normalize(answer) in normalize(context) for answer in answers))


def evaluate_remote_dataset(data_id: str, subset: str, metric: str = 'squad', split='train', token=None,
                            question_col: str = 'question', context_col: str = 'retrieved', predict_col: str = 'predicted',
                            id_col: str = 'question', label_col: str = 'answer', labeling: bool = True,
                            upload = False):
  dataset = load_dataset(data_id, subset)[split]
  score = evaluate_dataset(dataset, metric=metric, question_col=question_col,
                          context_col=context_col, predict_col=predict_col,
                          id_col=id_col, label_col=label_col, labeling=labeling)
  if upload:
    def replace_score(row, index):
      row['score'] = score[index]
      return row
    dataset = dataset.map(replace_score, with_indices=True)
    dataset.push_to_hub(data_id, subset, token=token)


def evaluate_dataset(dataset: Dataset, metric: str = 'squad',
                     question_col: str = 'question', context_col: str = 'retrieved', predict_col: str = 'predicted',
                     id_col: str = 'question', label_col: str = 'answer', labeling: bool = True):
  referee: QuestionAnsweringEvaluator = evaluator("question-answering")
  referee.PIPELINE_KWARGS["handle_impossible_answer"] = True
  metric_input, qa = referee.prepare_data(
      dataset, question_col, context_col, id_col, label_col)

  # References
  if labeling:
    for i, reference in enumerate(metric_input['references']):
      starts = [qa['context'][i].find(answer)
                for answer in reference['answers']]
      reference['answers'] = {
          'answer_start': starts, 'text': reference['answers']}

  # Prediction
  metric_input['predictions'] = []
  for row in list(dataset):
    result = {
        'prediction_text': row[predict_col], 'id': row[id_col]}
    if metric == 'squad_v2':
      result['no_answer_probability'] = 0.
    metric_input['predictions'].append(result)
  metric_module = referee.prepare_metric(metric)
  results = referee.compute_metric(metric_module, metric_inputs=metric_input)

  # Average tokens
  tokens = encoder.encode_batch(dataset['retrieved'])
  results['psgs_tokens'] = sum((len(token) for token in tokens)) / len(dataset)
  if dataset['summary'][0]:
    tokens = encoder.encode_batch(dataset['summary'])
    results['summary_tokens'] = sum((len(token)
                                    for token in tokens)) / len(dataset)

  # Summarizer
  if dataset['summary'][0]:
    contains = [exact_contain(row['answer'], row['retrieved'])
                for row in dataset]
    retains = [exact_contain(row['answer'], row['summary'])
               for row in dataset]
    contains_count = len(list(filter(lambda x: x, contains)))
    retains_count = len(list(filter(lambda x: x, retains)))
    # 답이 없었던 경우중 생긴 확률
    sum_tn = [True for contain, retain in zip(
        contains, retains) if not contain and retain]
    results['sum_tn'] = len(sum_tn) / len(dataset) * 100
    # 답이 있었던 경우중 없어진 확률
    sum_fn = [True for contain, retain in zip(
        contains, retains) if contain and not retain]
    # 원래 있던 것 중 원래 있는 확률
    sum_tp = [True for contain, retain in zip(
        contains, retains) if contain and retain]

    results['sum_rc'] = len(sum_tp) / (contains_count) * 100
    results['sum_fn'] = len(sum_fn) / len(dataset) * 100
    results['ret_em'] = contains_count / len(dataset) * 100
    results['sum_em'] = retains_count / len(dataset) * 100
  score = [exact_match(row[label_col], row[predict_col]) for row in dataset]
  results['exact_match'] = np.mean(score) * 100

  # Reader
  ctxs: List[str] = dataset['summary'] if dataset['summary'][0] else dataset['retrieved']
  contains = [exact_contain(row['answer'], ctxs[i])
              for i, row in enumerate(dataset)]
  matches = [exact_match(row['answer'], row['predicted'])
             != 0 for row in dataset]
  contains_count = len(list(filter(lambda x: x, contains)))
  # 답이 있는 경우중 틀릴 확률
  reader_fp = [True for contain, match in zip(
      contains, matches) if contain and not match]
  results['read_fp'] = len(reader_fp) / len(dataset) * 100
  # 답이 없는 경우중 맞출 확률
  reader_tn = [True for contain, match in zip(
      contains, matches) if not contain and match]
  results['read_tn'] = len(reader_tn) / len(dataset) * 100
  if 'ret_em' not in results:
    results['ret_em'] = contains_count / len(dataset) * 100

  print(len(dataset), results)
  return score


def normalize(s: str):
  """Normalize answer."""
  s = unicodedata.normalize("NFD", s)

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

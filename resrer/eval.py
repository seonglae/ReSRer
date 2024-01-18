import re
import string
from typing import List

import unicodedata
import tiktoken

from evaluate import evaluator, QuestionAnsweringEvaluator
from datasets import load_dataset, Dataset


def exact_match(answers, prediction):
  return any((normalize_answer(answer) == normalize_answer(prediction) for answer in answers))


def exact_contain(answers, context):
  return any((normalize_answer(answer) in normalize_answer(context) for answer in answers))


def evaluate_remote_dataset(id: str, subset: str, metric: str = 'squad', split='train',
                            question_col: str = 'question', context_col: str = 'retrieved', predict_col: str = 'predicted',
                            id_col: str = 'question', label_col: str = 'answer', labeling: bool = True):
  return evaluate_dataset(load_dataset(id, subset)[split], metric=metric, question_col=question_col,
                          context_col=context_col, predict_col=predict_col,
                          id_col=id_col, label_col=label_col, labeling=labeling)


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
  ctxs: List[str] = dataset['summary'] if dataset['summary'][0] else dataset['retrieved']
  encoder = tiktoken.encoding_for_model('gpt-4')
  tokens = encoder.encode_batch(ctxs)
  token_count = sum((len(token) for token in tokens)) / len(dataset)
  results['tokens'] = token_count

  # Reader
  contains = [exact_contain(row['answer'], ctxs[i]) for i, row in enumerate(dataset)]
  matchs = [exact_match(row['answer'], row['predicted']) for row in dataset]
  # 틀렸는데 있는경우
  reader_fp = [True for contain, match in zip(contains, matchs) if contain and not match]
  results['reader_fp'] = len(reader_fp) / len(dataset)
  # 없는데 맞춘경우
  reader_fn = [True for contain, match in zip(contains, matchs) if not contain and match]
  results['reader_fn'] = len(reader_fn) / len(dataset)
  reader_tp = [True for contain, match in zip(contains, matchs) if contain and match]
  results['reader_precision'] = len(reader_tp) / (len(reader_tp) + len(reader_fp))
  results['reader_recall'] = len(reader_tp) / (len(reader_tp) + len(reader_fn))

  # Summarizer
  if dataset['summary'][0]:
    contains = [exact_contain(row['answer'], row['retrieved'])
              for row in dataset]
    retains = [exact_contain(row['answer'], row['summary'])
                 for row in dataset]
    # 없었는데 생긴 경구
    sum_fp = [True for contain, retain in zip(contains, retains) if not contain and retain]
    results['sum_fp'] = len(sum_fp) / len(dataset)
    # 있었는데 없어진 경우
    sum_fn = [True for contain, retain in zip(contains, retains) if contain and not retain]
    results['sum_fn'] = len(sum_fn) / len(dataset)
    sum_tp = [True for contain, retain in zip(contains, retains) if contain and retain]
    results['sum_precision'] = len(sum_tp) / (len(sum_tp) + len(sum_fp))
    results['sum_recall'] = len(sum_tp) / (len(sum_tp) + len(sum_fn))
  return results


def evaluate_dataset_manual(id: str, subset: str):
  dataset = load_dataset(id, subset)
  dataset_list = list(dataset['train'])
  for row in dataset_list:
    row['score'] = max([regex_match_score(row['predicted'], answer)
                       for answer in row['answer']])
  score = sum([row['score'] for row in dataset_list]) / len(dataset_list)
  return score


def normalize_answer(s):
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


def exact_match_score(prediction, ground_truth):
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, ground_truth):
  try:
    regex = re.compile(ground_truth,
                       flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    return regex.match(prediction) is not None
  except re.error:
    return False

import re
import string

import unicodedata


from evaluate import evaluator, QuestionAnsweringEvaluator
from datasets import load_dataset


def evaluate_dataset(id: str, subset: str, metric: str = 'squad_v2',
                     question_col: str = 'question', context_col: str = 'retrieved', predict_col: str = 'predicted',
                     id_col: str = 'question', label_col: str = 'answer', labeling: bool = True):
  referee: QuestionAnsweringEvaluator = evaluator("question-answering")
  referee.PIPELINE_KWARGS["handle_impossible_answer"] = True

  # Dataset
  dataset = load_dataset(id, subset)
  dataset_list = list(dataset['train'])
  metric_input, _ = referee.prepare_data(
      dataset['train'], question_col, context_col, id_col, label_col)

  # References
  if labeling:
    for reference in metric_input['references']:
      reference['answers'] = {
          'answer_start': [0], 'text': reference['answers']}

  # Prediction
  metric_input['predictions'] = []
  for row in dataset_list:
    result = {
        'prediction_text': row[predict_col], 'id': row[id_col]}
    if metric == 'squad_v2':
      result['no_answer_probability'] = 0.
    metric_input['predictions'].append(result)

  metric_module = referee.prepare_metric(metric)
  results = referee.compute_metric(metric_module, metric_inputs=metric_input)
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

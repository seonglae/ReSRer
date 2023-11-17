from evaluate import evaluator
from datasets import load_dataset


def evaluate_dataset(id: str, subset: str, metric: str = 'squad_v2',
                     question_col: str = 'question', context_col: str = 'retrieved', predict_col: str = 'predicted',
                     id_col: str = 'question', label_col: str = 'answer', labeling: bool = True):
  referee = evaluator("question-answering")
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
        'prediction_text': row[predict_col], 'id': row[id_col], 'no_answer_probability': 0.}
    metric_input['predictions'].append(result)

  metric = referee.prepare_metric(metric)
  results = referee.compute_metric(metric, metric_inputs=metric_input)
  return results

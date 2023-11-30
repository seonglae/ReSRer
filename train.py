import fire
from dotenv import dotenv_values

from resrer.train import training

config = dotenv_values(".env")


def train(output='seonglae/resrer-bart', dataset_id='seonglae/resrer-nq', batch_size=8, epochs=4,
          checkpoint='facebook/bart-base', token=config['HF_TOKEN'], special_token=False):
  training(output=output, dataset_id=dataset_id, batch_size=batch_size, epochs=epochs,
           checkpoint=checkpoint, token=token, special_token=special_token)


if __name__ == '__main__':
  fire.Fire(train)

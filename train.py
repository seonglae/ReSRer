import fire

from resrer.train import training


def train(token):
  training(token=token)


if __name__ == '__main__':
  fire.Fire(train)

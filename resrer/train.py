from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import torch
from huggingface_hub import login

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def preprocesser(tokenizer):
  def preprocess_function(examples):
    inputs = [f"{examples['question_text'][i]}\n{doc}" for i,
              doc in enumerate(examples["document_text"])]
    model_inputs = tokenizer(inputs, truncation=True)
    labels = tokenizer(
        text_target=examples["summarization_text"], truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
  return preprocess_function


def training(output='seonglae/resrer-bart-base', dataset_id='seonglae/resrer-nq', checkpoint='facebook/bart-base',
             token=None):
  if token is not None:
    login(token=token)
  # Load model
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)

  # Load dataset
  dataset = load_dataset(dataset_id, split='train')
  splited_dataset = dataset.train_test_split(test_size=0.2)
  tokenized_dataset = splited_dataset.map(
      preprocesser(tokenizer), batched=True)
  print(tokenized_dataset["train"][0])

  # Train
  model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
  training_args = Seq2SeqTrainingArguments(
      output_dir=output,
      evaluation_strategy="epoch",
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      optim='adamw_hf',
      weight_decay=0.01,
      save_total_limit=3,
      num_train_epochs=4,
      push_to_hub=True,
  )
  trainer = Seq2SeqTrainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset["train"],
      eval_dataset=tokenized_dataset["test"],
      tokenizer=tokenizer,
      data_collator=data_collator,
  )
  trainer.train()

  # Push
  if token is not None:
    tokenizer.push_to_hub(f"{output}", token=token)
    model.push_to_hub(f"{output}", token=token)

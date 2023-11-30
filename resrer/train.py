from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
from huggingface_hub import login

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def preprocesser(tokenizer, special_token=False):
  def preprocess_function(examples):
    if special_token:
      inputs = [f"{examples['question_text'][i]}<sep>{doc}" for i,
                doc in enumerate(examples["document_text"])]
    else:
      inputs = [f"{examples['question_text'][i]}\n{doc}" for i,
                doc in enumerate(examples["document_text"])]
    model_inputs = tokenizer(inputs, truncation=True)
    labels = tokenizer(
        text_target=examples["summarization_text"], truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
  return preprocess_function


def training(output, dataset_id, checkpoint, batch_size=4, special_token=False,
             token=None, learning_rate=2e-5, weight_decay=0.01, epochs=4):
  if token is not None:
    login(token=token)
  # Load model
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)

  # Load dataset
  dataset = load_dataset(dataset_id, split='train')
  tokenized_dataset = dataset.map(
      preprocesser(tokenizer, special_token), batched=True)
  print(tokenized_dataset)

  # Train
  model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
  # Separator Token
  if special_token:
    tokenizer.add_tokens(['<sep>'])
    model.resize_token_embeddings(len(tokenizer))
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
  training_args = Seq2SeqTrainingArguments(
      output_dir=output,
      learning_rate=learning_rate,
      per_device_train_batch_size=batch_size,
      optim='adamw_hf',
      weight_decay=weight_decay,
      save_total_limit=3,
      num_train_epochs=epochs,
      push_to_hub=True,
  )
  trainer = Seq2SeqTrainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset,
      tokenizer=tokenizer,
      data_collator=data_collator,
  )
  trainer.train()

  # Push
  if token is not None:
    tokenizer.push_to_hub(f"{output}", token=token)
    model.push_to_hub(f"{output}", token=token)

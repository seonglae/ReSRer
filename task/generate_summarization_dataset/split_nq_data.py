"""
load simplified-nq-train.jsonl and split it into multiple chunks
"""

from typing import Dict

import datasets
from langchain.text_splitter import (
    # CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

INPUT_FILE_PATH = "./simplified-nq-train.jsonl"
OUTPUT_FILE_PATH = "./simplified-nq-split.parquet"


def check_answer_notempty(data: Dict) -> bool:
    return len(data["annotations"][0]["short_answers"]) != 0


# Dataset Code
def add_answer_to_data(data: Dict) -> Dict:
    # Extract document text
    document_text = data["document_text"].split()

    # Extract long answer start and end tokens
    long_answer = data["annotations"][0]["long_answer"]
    start_token, end_token = long_answer["start_token"], long_answer["end_token"]
    # Extract the long answer text
    long_answer_text = " ".join(document_text[start_token:end_token])
    data["long_answer_text"] = long_answer_text

    # Extract short answers
    short_answers = data["annotations"][0]["short_answers"][0]
    start_token, end_token = short_answers["start_token"], short_answers["end_token"]
    # Extract the short answer text
    short_answer_text = " ".join(document_text[start_token:end_token])
    data["short_answer_text"] = short_answer_text

    return data


def split_dataset_by_length_generator(
    dataset: datasets.Dataset, chunk_size: int = 4000, chunk_overlap=100
):
    for data in dataset:
        document = data["document_text"]
        answer = data["short_answer_text"]
        # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name="gpt-4"
        # )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(document)

        # yield the item for splitted dataset
        for i, chunk in enumerate(chunks):
            temp_data = data.copy()
            temp_data["document_text"] = chunk
            temp_data["split_id"] = f'{data["example_id"]}_{i + 1}'
            temp_data["answer_exist_chunk"] = answer in chunk

            yield temp_data


nq_dataset = datasets.Dataset.from_json(INPUT_FILE_PATH)

# nq_dataset = nq_dataset.select(range(100))  # for testing

nq_dataset = nq_dataset.filter(check_answer_notempty, num_proc=4)

nq_dataset = nq_dataset.map(add_answer_to_data, num_proc=4)

split_dataset = datasets.Dataset.from_generator(
    split_dataset_by_length_generator, gen_kwargs={"dataset": nq_dataset}, num_proc=4
)

split_dataset.to_parquet(OUTPUT_FILE_PATH)

"""
load splitted nq data and generate summarization dataset
"""

import asyncio
import argparse

from typing import Dict
from openai import OpenAI, AsyncOpenAI
from logging import getLogger

from tqdm import tqdm
import datasets
import pandas as pd

INPUT_FILE_PATH = "./simplified-nq-split.parquet"
COMPLETED_LOG_FILE_PATH = "./completed_files.txt"

MODEL = "gpt-3.5-turbo-0613"
API_KEY = ""

MAX_CONCURRENT_REQUESTS = 200  # Adjust as needed

client = AsyncOpenAI(api_key=API_KEY)
logger = getLogger(__name__)

parser = argparse.ArgumentParser(description="Generate summarization dataset")
parser.add_argument(
    "-s", "--start", dest="start", action="store", required=True, type=int
)  # extra value
parser.add_argument("-e", "--end", dest="end", action="store", required=True, type=int)

prompt_template = """\
You are a professional wikipedia writter.
Your task is to rewrite the given document to be easier for the reader to answer the given question.

The given document is a split of a article about the question topic.
The document is split into multiple parts, and this is one of them.
{answer_dependent_requirement}

Do not make up information that is not in the document, and do not answer the question.
It must be at least 400 words long.

<<Document>>
{document}
<<Question>>
{question}
<<Answer>>
{answer}

<<REMEMBER>>
KEEP THE STYLE OF THE ORIGINAL DOCUMENT IN YOUR SUMMARY. IT SHOULD BE WRITTEN LIKE IT's A WIKIPEDIA ARTICLE.
"""

in_case_of_non_existance = """The answer is not in the document, but you should rewrite the document anyway. Because the document is a split of a article about the question topic, and the answer is in the other split."""
in_case_of_existance = """The answer is in the document, so you should rewrite the document to make it easier for the reader to correctly answer the question."""


def generate_prompt(data: Dict) -> str:
    document = data["document_text"]
    question = data["question_text"]
    answer = data["short_answer_text"]

    answer_dependent_requirement = (
        in_case_of_existance if answer in document else in_case_of_non_existance
    )

    return prompt_template.format(
        document=document,
        question=question,
        answer=answer,
        answer_dependent_requirement=answer_dependent_requirement,
    )


def summarize_by_chatgpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL, temperature=0.0, messages=[{"role": "user", "content": prompt}]
    )
    print(response)
    return response.choices[0].message.content


async def summarize_by_chatgpt(prompt: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        timeout=30,
    )
    return response.choices[0].message.content


async def summarized_dataset_generator(data, semaphore: asyncio.Semaphore):
    async with semaphore:
        temp_data = data.copy()
        prompt = generate_prompt(temp_data)
        try:
            print("Processing split_id: ", data["split_id"])
            summarization = await summarize_by_chatgpt(prompt)
            print("done processing split_id: ", data["split_id"])
        except Exception as e:
            logger.error(
                "Error while generating summarization for split_id: %s",
                data["split_id"],
            )
            logger.error(e)
            summarization = ""
        finally:
            temp_data["summarization_text"] = summarization
        return temp_data


# get start and end range of the dataset
async def main():
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    args = parser.parse_args()
    processed_nq = datasets.Dataset.from_parquet(INPUT_FILE_PATH)
    processed_nq = processed_nq.select(
        range(max(args.start - 1, 0), args.end)
    )  # for testing
    print("Processing from {} to {}".format(args.start, args.end))

    # Process dataset asynchronously
    tasks = [summarized_dataset_generator(data, semaphore) for data in processed_nq]
    results = await asyncio.gather(*tasks)

    chatgpt_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=results))
    chatgpt_dataset.to_parquet(f"./data/chatgpt-nq_{args.start}-{args.end}.parquet")


asyncio.run(main())

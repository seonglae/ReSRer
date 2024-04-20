# ReSRer (Retriever, Summarizer, Reader)
Reducing context size and increasing QA score simultaneously for ODQA(Open-Domain Question Answering)


# Results
## Demo on Huggingface Space
- [Demo](https://huggingface.co/spaces/seonglae/resrer-demo) in Huggingface Space

<a href="https://huggingface.co/spaces/seonglae/resrer-demo">
  <img style="width: 75%" src="image/image.png" alt="ReSRer Demo" />
</a>




## Score results
[Total score resulst](https://huggingface.co/datasets/seonglae/nq_open-validation)

### Exact Match Increase Along Top-k Increase
<img style="width: 75%" src="https://github.com/seonglae/ReSRer/assets/27716524/ba5a6751-1091-498f-9807-ca431cb792d5" alt="ReSRer Demo" />

### Exact Match Shrinking Along QA Pipeline
<img style="width: 75%" src="https://github.com/seonglae/ReSRer/assets/27716524/82a15456-0cda-4a67-a2de-3ba3c3505fbb" alt="ReSRer Demo" />

### Token Count Changes Along Top-k Changing
<img style="width: 75%" src="https://github.com/seonglae/ReSRer/assets/27716524/3f518759-8687-4675-becf-c5df1d785651" alt="Token count" />


# Prompt
We mainly focused on NQ(Natural Question) dataset for this time.

## Reader prompt for NQ
```
Extract a concise noun-based answer from the provided context for the question. Your answer should be under three words and extracted directly from a context of no more than five words. You can analyze the context step by step to derive the answer. Avoid using prefixes that indicate the type of answer; simply present the shortest relevant answer span from the context.
```
## Summarizer prompt for NQ
We did several 
```
Condense the provided passages to focus on key elements directly answering the question. Your summary should be a third of the original passages' length and at least 150 words. Highlight critical information and evidence supporting the answer. Avoid generalizations or unrelated details. Ensure the final answer is present in the summary, keeping the exact span of the answer to under five words. Present the summary in a clear, bullet-point format for each key element related to the question. Aim for a balance between conciseness and completeness.
```

## Models
Trained model for ReSRer reader on Huggingface trained in [55k Training Dataset](https://huggingface.co/datasets/seonglae/resrer-nq) generated from GPT-3 with the below prompt
Our main goal was not to train a summarizing small model, but rather to prove that a summarizer module between the retriever and reader is an efficient method. So, we did not delve into training with the most recent summarizer prompt dataset. Therefore, this model's performance is not as good as with the original context (even better than native summarizer though). We disclose this because it might be helpful for people who want to reduce computing costs dramatically.
- [PegasusX](https://huggingface.co/seonglae/resrer-pegasus-x)
- [Bart](https://huggingface.co/seonglae/resrer-bart-base)


# Contribution
As I mentioned earlier, our research was aimed at exploring the potential benefits of an effective abstractive summarizer for QA tasks. Initially, we planned to test this approach. However, given the significant advancements made by SuRe and LLMLingua in this domain, we decided to halt our research.

Although our improvements of 4% (which translates to nearly 20% from the original score) may not seem impressive, we demonstrate that a single summarizer module can effectively handle simple tasks such as single-hop question answering, in contrast to more complex multi-step approaches. However, we were unable to confirm whether this single-step context pruning is effective for more intricate tasks like reasoning and code generation. Therefore, there may be room for further contributions in this area in the future.

# Get Started


### 1. Install dependencies

```bash
git clone https://github.com/seonglae/ReSRer
cd ReSRer
rye sync
# or
pip insatll .
# for training
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
pip install --force-reinstall typing-extensions==4.5.0
pip uninstall deepspeed
pip install deepspeed
pip uninstall -y apex
```

### 2. create .env

```bash
MILVUS_PW=
MILVUS_HOST=resrer
```

### 3. QA pipeline

```bash
python qa_pipeline.py
```

# Index to Vector DB

`indexing.json`

- check embedding dimension of tei
- subset target
- streaming or not
- collection name

```bash
python indexing.py
```

# TEI

[install guide](https://texonom.com/434f6f39b88342ea9e5156bd8501d8c4)

```
npm i -g pm2
model=
pm2 start data/tei.json
```

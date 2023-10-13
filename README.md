# ReSRer

## Introduction

## Get Started

### 1. Install dependencies

```bash
git clone https://github.com/seonglae/ReSRer
cd ReSRer
export export FAISS_ENABLE_GPU=ON
rye sync
# or
pip insatll .
```


### 2. Download Vector DB
```bash
git clone https://huggingface.co/datasets/seonglae/chroma_psgs_w100 data/chroma
```

### 3. QA pipeline
```bash
python qa_pipeline.py
```

# Index own DB

### 1. Download NQ data

You can find index id from the [source](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py)

```bash
git clone https://github.com/facebookresearch/DPR
# install deps
python data/download_data.py --resource indexes.single.nq.subset.index
python data/download_data.py --resource indexes.single.nq.subset.index_meta
# or indexes.single.nq.full
python data/download_data.py --resource data.wikipedia_split.psgs_w100
```

Move index files to `data/dpr/index` and context files to `data/dpr/ctx`

### 2. Migrate index data
```bash
python index_ctx.py
```

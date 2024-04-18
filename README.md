# ReSRer



# Results
## Models
Trained model for ReSRer reader in Huggingface
- [PegasusX](https://huggingface.co/seonglae/resrer-pegasus-x)
- [Bart](https://huggingface.co/seonglae/resrer-bart-base)

## Others
- [Training Dataset](https://huggingface.co/datasets/seonglae/resrer-nq) generated from GPT-3
- [Demo](https://huggingface.co/spaces/seonglae/resrer-demo) in Huggingface Space
[![ReSRer Demo](image/image.png)](https://huggingface.co/spaces/seonglae/resrer-demo)




## Score results
[Total score resulst](https://huggingface.co/datasets/seonglae/nq_open-validation)

![image](https://github.com/seonglae/ReSRer/assets/27716524/3f518759-8687-4675-becf-c5df1d785651)
![image](https://github.com/seonglae/ReSRer/assets/27716524/ba5a6751-1091-498f-9807-ca431cb792d5)
![image](https://github.com/seonglae/ReSRer/assets/27716524/82a15456-0cda-4a67-a2de-3ba3c3505fbb)




## Get Started


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

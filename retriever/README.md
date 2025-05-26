# Retrieval Model Training

## Installation

Installation from source

```bash
conda create -n retriever python==3.10
conda activate retriever
pip install -r requirements.txt
```
Download the corpus from the [RaCO](https://drive.google.com/drive/folders/1oj2POBBy8kyBFNU5nHb05wu2DlcOfGnV?usp=share_link) using the following command:
Unzip the file and run ```python merge-corpus.py``` to construct corpus

The retrieval model weights are in `/data/question_encoder/` or you can train them with the `train.sh`

Run the bash file with arguments for indexing the retrieval corpus.

```bash
bash create_index.sh

╭─ Arguments ────────────────────────────────────────────────────────────╮
│ *    retriever_path          TEXT  [retrival_model_path] [required]    │
│ *    data_path               TEXT  [splited_RaCO_path] [required]      │
│ *    output_dir              TEXT  [output_embedding_path] [required]  │
╰────────────────────────────────────────────────────────────────────────╯

```

# ReConnect: Retrieval-augmented Knowledge Connection for Commonsense Reasoning

## Install environment

```bash
conda create -n reconnect python=3.9
conda activate reconnect
pip install torch==2.1.1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run RALM

### Build commonsense retrieval corpus (i.e., RaCO) index
Download the corpus from the [RaCO](https://drive.google.com/drive/folders/1oj2POBBy8kyBFNU5nHb05wu2DlcOfGnV?usp=share_link) using the following command:
Unzip the file and run ```python merge-corpus.py``` to construct corpus

### Build commonsense dense retrieval corpus

Details are represented in README of retriever folder

```bash
cd retriever
bash create_index.sh
```

### Run

The parameters that can be selected in the config file `config.json` are as follows:

| parameter                 | meaning                                                      | example/options                                              |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `model_name_or_path`      | Hugging Face model.                                          | `meta-llama/Llama-2-13b-chat`                             |
| `method`                  | way to generate answers             | `non-retrieval`, `single-retrieval`, `zebra`, `ours` |
| `dataset`                 | Dataset                                                      | `csqa`, `csqa2`, `piqa`, `obqa`          |
| `zeroshot`                | Zeroshot.                                                    | true                                                            |
| `sample`                  | number of questions sampled from the dataset.<br />`-1` means use the entire data set. | 1000                                                         |
| `shuffle`                 | Whether to disrupt the data set.<br />Without this parameter, the data set will not be shuffled. | `true`, `false`(without)                                     |
| `generate_max_length`     | maximum generated length of a question                       | 1(fixed)                                                     |
| `retrieve_keep_top_k`     | number of reserved tokens when generating a search question  | 35                                                           |
| `output_dir`              | The generated results will be stored in a folder with a numeric name at the output folder you gave. If the folder you give does not exist, one will be created. | `../result/2wikimultihopqa_llama2_13b`                       |
| `retriever`               | type of retriever.                                           | `BM25`, `DPR`                                                |
| `retrieve_topk`           | number of related documents retained.                        | 3                                                            |

If you are using BM25 as the retriever, you should also include the following parameters

| Parameter       | Meaning                                    | example        |
| --------------- | ------------------------------------------ | -------------- |
| `es_index_name` | The name of the index in the Elasticsearch | `commonsense`  |


If you are using DPR as the retriever, you should also include the following parameters.

| Parameter                 | Meaning                               | example                                                   |
| ------------------------- | ------------------------------------- | --------------------------------------------------------- |
| `sgpt_model_name_or_path` | retrieval model (downloaded path)     | `/home/user10/RALM_CSQA/retriever/question_encoder`       |
| `embedding_path`          | retrieval embedding folder path       | `/home/user10/RALM_CSQA/documents`                        |
| `passage_file`            | Path to the Wikipedia dump            | `None`                                                    |

Here is the config file for using our approach to generate answers to the top 3000 questions of CSQA using the model Llama-3.1-8b-chat.

```json
{
    "model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "method": "ours",
    "dataset": "csqa",
    "data_path": "../data/csqa",
    "fewshot": 0,
    "sample": 3000,
    "shuffle": false,
    "generate_max_length": 1,
    "query_formulation": "direct",
    "output_dir": "../result/llama3_8b_chat_csqa_zeroshot",
    "retriever": "DPR",
    "es_index_name": "commonsense",
    "retrieve_topk": 3,
    "use_counter": true,
    "load" : false,
    "zeroshot" : true,
    "retrieval_model_name_or_path" : "/home/user10/RALM_CSQA/retriever/question_encoder",
    "embedding_path" : "/home/user10/RALM_CSQA/documents"
}
```

The config files of the main experiments in the paper are all in the `config/`.

When you have prepared the configuration file, run the following command in the `src` directory:

```bash
cd src
bash /home/user10/RALM_CSQA/src/generation.sh

```



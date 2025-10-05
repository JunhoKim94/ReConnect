# ReConnect: Retrieval-augmented Knowledge Connection for Commonsense Reasoning (EMNLP 2025)

## Install environment

```bash
conda create -n reconnect python==3.10
conda activate reconnect
pip install torch==2.1.1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run ReConnect

### Build commonsense retrieval corpus (e.g., RaCO, COCONUT, ZEBRA) index
Download the corpus from the [RaCO](https://drive.google.com/drive/folders/1oj2POBBy8kyBFNU5nHb05wu2DlcOfGnV?usp=share_link) using the following command:
Unzip the file and run ```python merge-corpus.py``` to construct corpus

### Build commonsense dense retrieval corpus
Details are represented in README of retriever folder

```bash
cd retriever
bash create_index.sh
```

We construct dummy document files, which are sampled from original retrieval corpus, to run our ReConnect. We will release our full retrieval corpus. 

### Run

The parameters that can be selected in the config file `config.json` are as follows:

| parameter                 | meaning                                                      | example/options                                              |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `model_name_or_path`      | Hugging Face model.                                          | `meta-llama/Llama-2-13b-chat`                             |
| `method`                  | way to generate answers             | `non-retrieval`, `single-retrieval`, `zebra`, `reconnect` |
| `dataset`                 | Dataset                                                      | `id` `ood`          |
| `zeroshot`                | Zeroshot.                                                    | true                                                            |
| `sample`                  | number of questions sampled from the dataset.<br />`-1` means use the entire data set. | 1000                                                         |
| `shuffle`                 | Whether to disrupt the data set.<br />Without this parameter, the data set will not be shuffled. | `true`, `false`(without)                                     |
| `generate_max_length`     | maximum generated length of a question                       | 1(fixed)                                                     |
| `output_dir`              | The generated results will be stored in a folder with a numeric name at the output folder you gave. If the folder you give does not exist, one will be created. | `../result/`                       |
| `retriever`               | type of retriever.                                           | `BM25`, `DPR` `DPR_Ours` `DPR_ZEBRA`                        |
| `retrieve_topk`           | number of related documents retained.                        | 5                                                            |


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
    "retriever": "DPR_Ours",
    "retrieve_topk": 5,
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



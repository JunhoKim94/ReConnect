from argparse import ArgumentParser

import os
import jsonlines
import torch
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from tqdm import tqdm

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): Command-line arguments.
    """
    parser = ArgumentParser()

    # Add arguments.
    parser.add_argument(
        "--data_paths",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to the dataset(s) to be included in the document index.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the document index.",
    )

    # Add optional arguments.
    parser.add_argument(
        "--retriever_path",
        type=str,
        default=RETRIEVER_PATH,
        help="Path to the retriever model. Can be either a local path or a model ID from Hugging Face.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for encoding the documents.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for encoding the documents.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of the input sequences.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: 'cuda').",
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="fp32", 
        choices=["fp32", "fp16"],
        help="Precision.")

    # Parse arguments.
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    documents = []
    
    label2alpabet = {0: "A", 1:"B", 2: "C", 3: "D", 4: "E", 5: "F", 6 : "G", 7 : "H"}
    
    for dataset in args.data_paths:
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"The data path {dataset} does not exist. Please provide a valid path.")
        with jsonlines.open(dataset, "r") as current_dataset:
            for sample in tqdm(current_dataset):
                #print(sample)
                id = sample["idx"]
                
                question = sample["question"]
                #question = sample["text"]
                #question = sample["metadata"]["question"]
                #choices = sample["metadata"]["candidate"]
                #joined_choices =  " [SEP] ".join(
                #    [f"{label2alpabet[i]}. {choice}" for i, choice in enumerate(choices)]
                #)
                #passage = f"{question} [SEP] {joined_choices}"
                
                #question = sample["question"]["stem"]
                #choices = sample["question"]["choices"]
                #metadata = {"answerKey": sample["answerKey"]}
                #joined_choices = " [SEP] ".join(
                #    [f"{choice['label']}. {choice['text']}" for choice in choices]
                #)
                #passage = f"{question} [SEP] {joined_choices}"
                passage = question
                #documents.append({"text": passage, "id": id, "metadata": metadata})
                documents.append({"text": passage, "id": id})
                
    documents = DocumentStore.from_dict(documents)

    document_index = InMemoryDocumentIndex(
        documents=documents,
        device=args.device,
        precision=args.precision,
    )

    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder=args.retriever_path,
        document_index=document_index,
        device=args.device,
    )

    retriever.index(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        force_reindex=True,
        precision=args.precision,
    )

    retriever.document_index.save_pretrained(output_dir=args.output_dir)
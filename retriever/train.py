from argparse import ArgumentParser
from goldenretriever import GoldenRetriever, Trainer
from goldenretriever.common.log import get_logger
from goldenretriever.data.datasets import InBatchNegativesDataset
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex

logger = get_logger(__name__)

def parse_arguments():
    arg_parser = ArgumentParser(
        description="Train a retriever model for Zebra"
    )
    arg_parser.add_argument(
        "--question_encoder", type=str, default="intfloat/e5-base-v2"
    )
    arg_parser.add_argument("--passage_encoder", type=str, default=None)
    arg_parser.add_argument("--document_index", type=str, default=None)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--precision", type=str, default="16")
    # data
    arg_parser.add_argument("--train_data_path", type=str, required=True)
    arg_parser.add_argument("--dev_data_path", type=str, required=True)
    arg_parser.add_argument("--question_batch_size", type=int, default=64)
    arg_parser.add_argument("--passage_batch_size", type=int, default=200)
    arg_parser.add_argument("--max_question_length", type=int, default=256)
    arg_parser.add_argument("--max_passage_length", type=int, default=256)
    
    # train
    arg_parser.add_argument("--max_steps", type=int, default=25_000)
    arg_parser.add_argument("--num_workers", type=int, default=0)
    arg_parser.add_argument("--max_hard_negatives_to_mine", type=int, default=0)
    arg_parser.add_argument("--wandb_online_mode", action="store_true")
    arg_parser.add_argument("--wandb_log_model", action="store_true")
    arg_parser.add_argument("--wandb_project_name", type=str, default="zebra-retriever")
    arg_parser.add_argument(
        "--wandb_experiment_name",
        type=str,
        default="ours_explanations",
    )
    
    return arg_parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder=args.question_encoder,
        passage_encoder=args.passage_encoder,
        document_index=InMemoryDocumentIndex(
            documents=(
                DocumentStore.from_file(args.document_index)
                if args.document_index
                else DocumentStore()
            ),
            device=args.device,
            precision=args.precision,
        ),
    )
    
    print(args.train_data_path)

    train_dataset = InBatchNegativesDataset(
        name="raco_train",
        path=args.train_data_path,
        tokenizer=retriever.question_tokenizer,
        question_batch_size=args.question_batch_size,
        passage_batch_size=args.passage_batch_size,
        max_question_length=args.max_question_length,
        max_passage_length=args.max_passage_length,
        shuffle=True,
    )
    dev_dataset = InBatchNegativesDataset(
        name="raco_dev",
        path=args.dev_data_path,
        tokenizer=retriever.question_tokenizer,
        question_batch_size=args.question_batch_size,
        passage_batch_size=args.passage_batch_size,
        max_question_length=args.max_question_length,
        max_passage_length=args.max_passage_length,
    )

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=dev_dataset,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        wandb_online_mode=args.wandb_online_mode,
        wandb_log_model=False,
        wandb_project_name=args.wandb_project_name,
        wandb_experiment_name=args.wandb_experiment_name,
        max_hard_negatives_to_mine=args.max_hard_negatives_to_mine,
    )

    trainer.train()
    

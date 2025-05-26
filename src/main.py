import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from generate import BasicRAG, SingleRAG, OurRAG, ZEBRA
import torch
import random
import pickle
from data import OOD_Dataset, ID_Dataset

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--query_expansion", action='store_true')
    parser.add_argument("--knowledge_generation", action='store_true')
    parser.add_argument("--knowledge_aggregation", action='store_true')
    
    parser.add_argument("--retrieve_topk", type = int, default = 3)
    parser.add_argument("--dataset", type = str, default = "id")
    parser.add_argument("--augmentation", action='store_true')
    parser.add_argument("--num_knowledge", type = int, default=3)
    parser.add_argument("--sampling", type = str, default="reconnect")
    
    parser.set_defaults(query_expansion=False)
    parser.set_defaults(knowledge_generation=False)        
    parser.set_defaults(augmentation=False)
    parser.set_defaults(knowledge_aggregation=False)
    

    args = parser.parse_args()
    config_path = args.config_path
    args_list = vars(args)
    with open(config_path, "r") as f:
        temp = json.load(f)
        for k,v in temp.items():
            if isinstance(v, bool):
                v = str(v).lower()
            
            if k in args_list.keys():
                continue
            parser.add_argument(f'--{k}', type=type(v), default=v)
            

    args = parser.parse_args()

    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    if "zeroshot" not in args: 
        args.zeroshot = False
    if "augmentation" not in args:
        args.augmentation = False

    print(args.load, args.query_expansion, args.knowledge_generation)

    return args

def get_model_answer(
    log_prob,
    tokenizer,
    labels,
    return_scores=False,
):
    """
    Get the answer from the model.

    Parameters:
    - tokenizer (AutoTokenizer): Tokenizer.
    - labels (Optional[List[str]]): Labels, default is ["A", "B", "C", "D", "E"].
    - return_scores (Optional[bool]): Return scores.

    Returns:
    - str: Answer (one of the labels).
    - Optional[torch.Tensor]: Scores (if return_scores is True).
    """
    # Get the probabilities of the first token.
    probabilities = log_prob

    # Check that the labels are in the tokenizer's vocabulary.
    labels = [label for label in labels if len(tokenizer.tokenize(label)) == 1]

    # Get the label IDs.
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]

    # Get the probability of each label (A, B, C, D, E) and its variants.
    answer = [probabilities[label_id].item() for label_id in label_ids]

    # Get the label with the highest probability.
    answer = labels[answer.index(max(answer))]
    answer = answer.lstrip()

    if return_scores:
        return answer, probabilities
    else:
        return answer

def main():

    args = get_args()
    logger.info(f"{args}")

    # load data
    #sampling_strategy = ["deterministic", "clustering", "random", "reconnect"]
    sampling_strategy = ["reconnect"]

    if args.dataset == "id":
        data_list = ["obqa", "wg", "csqa", "arc-challenge", "arc-easy", "piqa", "csqa2", "qasc"]
        batch_size = 8
    elif args.dataset == "ood":
        data_list = ["numersense", "quartz", "sct", "riddlesense", "sciq", "wsc", "quarel", "hellaswag"]
        batch_size = 8
        
    if args.method == "non-retrieval":
        model = BasicRAG(args, [None, None, None, None])
    elif args.method == "single-retrieval":
        model = SingleRAG(args, [None, None, None, None])
    elif args.method == "reconnect":
        print('our rag')
        model = OurRAG(args, "", [None, None, None, None, None, None])
    elif args.method == "zebra":
        model = ZEBRA(args, "", [None, None, None, None])
    else:
        raise NotImplementedError

    for sampling in sampling_strategy:

        args.sampling = sampling
        model.sampling = sampling
        
        for d in data_list:
            args.dataset = d
            args.output_dir = "../result/%s_%s_zeroshot/"%(args.model_name_or_path.split("/")[-1], d)
            
            # output dir
            if os.path.exists(args.output_dir) is False:
                os.makedirs(args.output_dir)
            dir_name = os.listdir(args.output_dir)
            for i in range(10000):
                if (args.method + "_%d"%i) not in dir_name:
                    args.output_dir = os.path.join(args.output_dir, args.method + "_%d"%i)
                    os.makedirs(args.output_dir)
                    break
            logger.info(f"output dir: {args.output_dir}")
            # save config
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            # create output file
            output_file = open(os.path.join(args.output_dir, "output.txt"), "w")
            output_file2 = open(os.path.join(args.output_dir, "output_explanation.txt"), "w")

            # load data
            if args.dataset in ["csqa", "csqa2", "piqa", "arc", "obqa", "qasc", "wg", "arc-challenge", "arc-easy"]:
                data = ID_Dataset(args, args.data_path)    
            elif args.dataset in ["hellaswag","riddlesense","numersense", "quartz", "quarel", "sciq", "wsc", "sct"]:
                data = OOD_Dataset(args, args.data_path)
            else:
                raise NotImplementedError

            model.inst = data.inst
            model.reply = data.reply
            model.answer = data.answer
            model.knowledge_inst = data.knowledge_inst
            
            if "reconnect" in args.method:
                print(args.method)
                model.knowledge_inst_external = data.knowledge_inst_external
                model.knowledge_selection = data.knowledge_selection
                model.knowledge_refinement = data.knowledge_refinement
                
            model.choice_num = data.choice_num
            labels = data.labels

            # Generate alternative labels with whitespaces in front.
            labels.extend([f" {label}" for label in labels])

            data.format(fewshot=args.fewshot)
            data = data.dataset
            print(len(data))

            logger.info("start inference")
            ans_list = []
            gt_ans_list = []
            
            for i in tqdm(range(len(data) // batch_size), ncols = 100):
                last_counter = copy(model.counter)
                batch = data[batch_size * i : batch_size * (i+1)]
                #print(batch)

                if args.zeroshot:
                    #print("Zeroshot method")
                    
                    if "reconnect" in args.method:               
                        pred, selected_docs, quries, log_prob, explanations, docs = model.inference_batch(batch["question"], batch["choice"], query_expansion= args.query_expansion, knowledge_generation = args.knowledge_generation, knowledge_aggregation = args.knowledge_aggregation, N = args.num_knowledge)
                    elif "zebra" in args.method:
                        pred, docs, quries, log_prob, explanations = model.inference_batch(batch["question"], batch["choice"], k = args.retrieve_topk)
                        selected_docs = [None] * batch_size
                    else:
                        pred, docs, quries, log_prob = model.inference_batch(batch["question"], batch["choice"])
                        
                        explanations = [None] * len(batch)
                        docs = [None] * batch_size
                        exp_history = explanations
                        
                    pred_ans = [get_model_answer(prob, model.generator.tokenizer, labels) for prob in log_prob]
                    ans_list += pred_ans
                    gt_ans_list += batch["answer"]

                else:
                    print("Other method")
                    pred, docs, quries, _ = model.inference_batch(batch["question"], batch["demo"], batch["case"], batch_decode = True)
                    docs = [None] * len(batch)

                #print(docs)

                for j in range(batch_size):
                    if args.method == "non-retrieval" or args.method == "single-retrieval":
                        ret = {
                            'question' : batch["question"][j],
                            "prediction": pred[j].strip(),
                            "docs" : str(docs[j]),
                            "answer" : batch["answer"][j],
                            "qid": batch["qid"][j],
                        }
                    else:
                        #print(explanations, j)
                        ret = {
                            "prediction": str(pred[j]),
                            "answer" : batch["answer"][j],
                            "quries" : str(quries[j]),
                            "explain" : str(explanations[j]),
                            "selected_docs" : str(selected_docs[j]),
                            "docs" : str(docs[j]),
                            'question' : batch["question"][j],
                            #'choice' : batch["choice"][j]
                            #"qid": batch["qid"][j],
                        }
                        
                    if "reconnect" in args.method:
                        ret2 = {
                            "explain" : str(explanations[j]),
                            "selected_docs" : str(selected_docs[j]),
                            "docs" : str(docs[j])
                        }
                        output_file2.write(json.dumps(ret2) + "\n")

                            
                    output_file.write(json.dumps(ret)+"\n")


            if args.zeroshot:
                result_file = open(os.path.join(args.output_dir, "result.txt"), "w")

                score = 0
                for p_a, g_a in zip(ans_list, gt_ans_list):
                    if p_a == g_a:
                        score += 1
                
                score /= len(ans_list)
                print(f"\nAcc : {score}\n")
                
                result_file.write(f"\nAcc : {score}\n")
                result_file.close()

                with open("./total_results.txt","a") as f:
                    if "retriever" not in args:
                        retriever = None
                    else:
                        retriever = args.retriever
                    f.write(os.path.join(args.output_dir, args.method + f"_{i}_{args.query_expansion}_{args.knowledge_generation}_{args.knowledge_aggregation}_{args.retrieve_topk}_{retriever}_{args.num_knowledge}_{args.sampling}") + f" | Acc : {score}\n")
                    
            output_file.close()
            output_file2.close()


if __name__ == "__main__":
    main()
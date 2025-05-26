from typing import Dict, List, Callable, Tuple, Union, Callable
import logging
import os
import json
import re
import glob
import string
import spacy
from collections import Counter
from tqdm import tqdm
import numpy as np
from datasets import Dataset, load_dataset
import random

from prompt.prompts import DATASET_TAGS, SHOT_TEMPLATES

#from zebra.our_prompts import OUR_DATASET_TAGS
#from zebra.our_prompts_direct import OUR_DATASET_TAGS
#from zebra.our_prompts_ckd import OUR_DATASET_TAGS

#from zebra.reconnect import OUR_DATASET_TAGS
from prompt.reconnect import OUR_DATASET_TAGS


#from zebra.our_prompts_rs import OUR_DATASET_TAGS
            
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BaseDataset:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        return {}

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


    def format(self, fewshot: int = 0):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            
            #if self.args.zeroshot:
            #case = [self.inst, self.reply, case, self.answer]
            #case = self.tokenizer.apply_chat_template(case)
        
            # ctx
            example['demo'] = demo
            example['case'] = case
            #example["choice"] = 
            
            return example
        self.dataset = self.dataset.map(_format_for_dataset)
    
    def get_real_prediction(self, pred):
        return pred

class ID_Dataset(BaseDataset):
    def __init__(self, args, data_path: str):
        self.args = args
        logger.info(f"Loading {args.dataset} from {data_path}")
        dataset = []
        with open("/home/user10/ReConnect/data/ID/%s/%s-dev.jsonl"%(args.dataset, args.dataset), "r") as f:
            test_dataset = f.readlines()
            #dataset_1 = json.load(fz
            #test_dataset = json.load(f)

        if args.dataset == "csqa":
            self.labels = ["A","B","C","D", "E"]
        elif args.dataset == "csqa2":
            self.labels = ["A", "B"]
        elif args.dataset == "piqa":
            self.labels = ["A", "B"]
        elif args.dataset == "obqa":
            self.labels = ["A", "B", "C", "D"]
            with open("/home/user10/ReConnect/data/ID/%s/%s-test.jsonl"%(args.dataset, args.dataset), "r") as f:
                test_dataset = f.readlines()
        elif args.dataset == "qasc":
            self.labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        elif args.dataset == "wg":
            self.labels = ["A", "B"]
        elif "arc" in args.dataset:
            self.labels = ["A", "B", "C", "D"]  
            with open("/home/user10/ReConnect/data/ID/%s/%s-test.jsonl"%(args.dataset, args.dataset), "r") as f:
                test_dataset = f.readlines()

        self.choice_num = len(self.labels)
        self.demo_input_template = lambda ques: f'Question: {ques}\nAnswer:'

        if self.args.method == "reconnect":
            self.inst = OUR_DATASET_TAGS[args.dataset]["mcq_with_kg"]
            self.knowledge_inst = OUR_DATASET_TAGS[args.dataset]["knowledge_generation"]
            
            self.knowledge_inst_external = OUR_DATASET_TAGS[args.dataset]["knowledge_generation_external"]
            self.knowledge_selection = OUR_DATASET_TAGS[args.dataset]["knowledge_selection"]
            
            self.knowledge_refinement = OUR_DATASET_TAGS[args.dataset]["knowledge_refinement"]
            
            #self.summarization_inst = None
            self.reply = 'Yes, I understand. Please provide the question and the possible options.'
            self.test_input_template = lambda ques: f'{ques}'
            self.answer = "Answer:"

        else:
            self.inst = DATASET_TAGS[args.dataset]["mcq_with_kg"]
            self.knowledge_inst = DATASET_TAGS[args.dataset]["knowledge_generation"]
            self.reply = 'Yes, I understand. Please provide the question and the possible options.'
            self.test_input_template = lambda ques: f'{ques}'
            self.answer = "Answer:"

            
        self.output_template = lambda cot, ans: f'{cot} So the answer is {ans}.'

        idx = 0
        for data in tqdm(test_dataset):
            data = json.loads(data)
            #print(data)
            question = data["question"]["stem"]
            ans = data["answerKey"]
            
            choices = data["question"]["choices"]
            
            example = {
                "qid": idx, 
                "question": question, 
                #"ctxs" : data["texts"],
                #"cot": " ".join(data["facts"]), 
                "choice" : choices,
                "answer" : ans
            }
            idx += 1
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)
        
        print(len(self.dataset))

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                if pred.endswith("</s>"):
                    pred = pred[:len(pred) - len("</s>")]
                if pred.endswith("<|endoftext|>"):
                    pred = pred[:len(pred) - len("<|endoftext|>")]
                if pred.endswith("."):
                    pred = pred[:-1]
                    
                return pred
        else:
            return ""



class OOD_Dataset(BaseDataset):
    def __init__(self, args, data_path: str):
        self.args = args
        logger.info(f"Loading CSQA from {data_path}")
        dataset = []
        #with open("/home/user10/RALM_CSQA/data/%s/dev_2.jsonl"%args.dataset, "r") as f:
        with open("/home/user10/ReConnect/data/OOD/%s-dev.jsonl"%args.dataset, "r") as f:
            test_dataset = f.readlines()

        if args.dataset == "hellaswag":
            self.labels = ["A","B","C","D"]
        elif args.dataset == "siqa":
            self.labels = ["A", "B", "C"]
        elif args.dataset == "riddlesense":
            self.labels = ["A", "B", "C", "D", "E"]
        elif args.dataset == "com2sense":
            self.labels = ["A", "B"]
        elif args.dataset == "numersense":
            self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        elif args.dataset == "quartz":
            self.labels = ["A", "B"]
        elif args.dataset == "quarel":
            self.labels = ["A", "B"]
        elif args.dataset == "prost":
            self.labels = ["A","B","C","D"]
        elif args.dataset == "sciq":
            self.labels = ["A","B","C","D"]
        elif args.dataset == "wsc":
            self.labels = ["A", "B"]
        elif args.dataset == "cycic":
            self.labels = ["A", "B", "C", "D", "E"]
        elif args.dataset == "sct":
            self.labels = ["A", "B"]

        self.choice_num = len(self.labels)
        self.demo_input_template = lambda ques: f'Question: {ques}\nAnswer:'

        if self.args.method == "reconnect":
            self.inst = OUR_DATASET_TAGS[args.dataset]["mcq_with_kg"]
            self.knowledge_inst = OUR_DATASET_TAGS[args.dataset]["knowledge_generation"]
            
            self.knowledge_inst_external = OUR_DATASET_TAGS[args.dataset]["knowledge_generation_external"]
            self.knowledge_selection = OUR_DATASET_TAGS[args.dataset]["knowledge_selection"]
            self.knowledge_refinement = OUR_DATASET_TAGS[args.dataset]["knowledge_refinement"]

            self.reply = 'Yes, I understand. Please provide the question and the possible options.'
            self.test_input_template = lambda ques: f'{ques}'
            self.answer = "Answer:"

        else:
            self.inst = DATASET_TAGS[args.dataset]["mcq_with_kg"]
            self.knowledge_inst = DATASET_TAGS[args.dataset]["knowledge_generation"]
            self.reply = 'Yes, I understand. Please provide the question and the possible options.'
            self.test_input_template = lambda ques: f'{ques}'
            self.answer = "Answer:"

            
        self.output_template = lambda cot, ans: f'{cot} So the answer is {ans}.'

        idx = 0
        for data in tqdm(test_dataset):
            data = json.loads(data)
            #print(data)
            question = data["question"]["stem"]
            ans = data["answerKey"]
            
            choices = data["question"]["choices"]
            
            example = {
                "qid": idx, 
                "question": question, 
                #"ctxs" : data["texts"],
                #"cot": " ".join(data["facts"]), 
                "choice" : choices,
                "answer" : ans
            }
            idx += 1
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)
        
        print(len(self.dataset))

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                if pred.endswith("</s>"):
                    pred = pred[:len(pred) - len("</s>")]
                if pred.endswith("<|endoftext|>"):
                    pred = pred[:len(pred) - len("<|endoftext|>")]
                if pred.endswith("."):
                    pred = pred[:-1]
                    
                return pred
        else:
            return ""


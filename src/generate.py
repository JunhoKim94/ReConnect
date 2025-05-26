import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, DPR_ZEBRA, DPR, DPR_Single, DPR_Test
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from evaluate import load
from model import GoldenRetrieverModel
import copy

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans

import random

from prompt.prompt_utils import (prepare_sample_for_knowledge_generation, prepare_sample_for_mcq)
from prompt.reconnect import SHOT_TEMPLATES

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
TOKEN = "Use Your Huggingface Token"

print(torch.cuda.is_available())
device = torch.device("cuda")

class BasicGenerator:
    def __init__(self, model_name_or_path, load = 0):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token = TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2",
                    trust_remote_code = "falcon" in model_name_or_path, use_auth_token = TOKEN)

        self.model.eval()
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left" 
        #self.model.to(device)

        self.space_token = self.tokenizer.tokenize(' ')[0]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False, batch_decode = False):
        if batch_decode:
            input_ids = self.tokenizer(input_text, return_tensors="pt", padding = True)
            attention_mask = input_ids["attention_mask"].to(self.device)
            input_ids = input_ids["input_ids"]
        else:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            attention_mask = torch.ones_like(input_ids).to(self.device)

        input_ids = input_ids.to(self.device)
        input_length = input_ids.shape[1]

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                do_sample=False,
                #temperature=0.0,
                top_p=1.0,
                num_beams=1,
                output_scores = True,
            )

            generated_tokens = outputs.sequences[:, input_length:]
            if batch_decode:
                text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
                tokens = []
                for sentence in generated_tokens:
                    tokens += [self.tokenizer.decode(t, skip_special_tokens = True) for t in sentence]

                logprobs = torch.softmax(outputs.scores[-1], dim=-1)#[0]
                logprobs = logprobs.cpu().numpy()
            else:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )

                text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens = True)
                tokens = [self.tokenizer.decode(t, skip_special_tokens = True) for t in generated_tokens[0]]
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]

            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )

            generated_tokens = outputs[:, input_length:]
            
            if batch_decode:
                text = self.tokenizer.batch_decode(generated_tokens)
            else:
                text = self.tokenizer.decode(generated_tokens[0])
            
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()

        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())

            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    qa_input_template = lambda self, ques: f'Question: {ques} \n\n Answer:'
    def __init__(self, args, templates = None):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        print(args)
        self.retrieval_model_name_or_path = args['retrieval_model_name_or_path']

        self.generator = BasicGenerator(self.model_name_or_path, args["load"])
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                print(self.es_index_name)
                
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )

            elif self.retriever_type == "DPR_ZEBRA":
                self.retriever = DPR_ZEBRA(
                    model_name_or_path = self.retrieval_model_name_or_path, 
                    sgpt_encode_file_path = None,
                    passage_file = None,
                    )

            elif self.retriever_type == "DPR_Reconnect":
                self.retriever =  DPR(
                    retrieval_model_name_or_path = self.retrieval_model_name_or_path, 
                    embedding_path=self.retrieval_embedding_path,
                    passage_file=None
                )
                
            elif self.retriever_type == "DPR_Origin":
                self.retriever = DPR(
                    retrieval_model_name_or_path="intfloat/e5-base-v2",
                    embedding_path=self.retrieval_embedding_path,
                    passage_file=None
                )

            elif self.retriever_type == "DPR_Test":
                self.retriever = DPR_Test(
                    retrieval_model_name_or_path = self.retrieval_model_name_or_path, 
                    embedding_path=self.retrieval_embedding_path,
                    passage_file=None
                )
            else:
                raise NotImplementedError
        
        self.counter = Counter()
        self.templates = templates
        
        self.device = torch.device("cuda:0")
        self.generator.model.to(self.device)

        if templates != None:
            self.inst = templates[0]
            self.reply = templates[1]
            self.answer = templates[2]
            self.knowledge_inst = templates[3]

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]

        elif "DPR" in self.retriever_type:
            docs, distances = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            #return np.array(docs[0]) 
            return docs[0], distances
        
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        
        #print(prompt)
        
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, None, None
    
    def tokenizer_handler(
            self,
            inputs
        ):
        inputs = inputs[:, :-1]
        return inputs
    
    def generate_text(
        self,
        prompt_list,
        max_new_tokens = 256):
        """
        Generate text using the model.

        Parameters:
        - model (AutoModelForCausalLM): Model.
        - tokenizer (AutoTokenizer): Tokenizer.
        - prompt (List[str]): List of conversation turns.
        - max_new_tokens (int): Maximum number of new tokens.
        - device (str): Device to run the model on (default is 'cuda' if available, otherwise 'cpu').

        Returns:
        - tuple: A tuple containing the model outputs and the generated text.
        """
        # Build the conversation turns in the format required by chat templates.
        
        input_list = []
        #attention_mask = []
        
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]            
            messages = []
            for turn_id, turn in enumerate(prompt):
                if turn_id % 2 == 0:
                    messages.append({"role": "user", "content": turn})
                else:
                    messages.append({"role": "assistant", "content": turn})

            inputs = self.generator.tokenizer.apply_chat_template(messages, return_tensors="pt")#.to(device)
            inputs = self.tokenizer_handler(inputs)
        
            input_list.append(inputs[0])
        
        
        max_length = [len(s) for s in input_list]
        max_length = max(max_length)
        
        input_ids = torch.zeros((len(input_list), max_length)).to(torch.long).fill_(self.generator.tokenizer.pad_token_id)
        attention_mask = torch.zeros((len(input_list), max_length)).to(torch.long)
        
        for idx in range(len(input_list)):
            length = len(input_list[idx])
            #print(input_list[idx])
            
            input_ids[idx, -length:] = input_list[idx]
            attention_mask[idx, -length:] = 1
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        #attention_mask = torch.ones_like(inputs).to(device)
        
        outputs = self.generator.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens, 
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            return_dict_in_generate = True, 
            output_scores = True,
        )

        input_length = input_ids.shape[1]

        generated_tokens = outputs.sequences[:, input_length:]

        text = self.generator.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
        #text = [self.tokenizer.decode(t) for t in generated_tokens]
        tokens = []
        for sentence in generated_tokens:
            tokens += [self.generator.tokenizer.decode(t, skip_special_tokens = True) for t in sentence]

        logprobs = torch.softmax(outputs.scores[-1], dim=-1)#[0]
        logprobs = logprobs.cpu().numpy()

        
        return text, tokens, logprobs
    
    def inference_batch(self, question, choices):
        # non-retrieval
        prompt_list = []
        for idx in range(len(question)):
            #prompt = "".join([d["case"]+"\n" for d in demo[i]])
            #prompt += case[i]
            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            prompt = [self.inst, self.reply]
            final_shot = SHOT_TEMPLATES["mcq"].format(question=q, choices=c)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            
            prompt_list.append(prompt)
    
        #if log_prob:
        text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        #else:
        #    text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        
        if self.use_counter == True:
            for i in range(len(question)):
                self.counter.add_generate(text[i], self.generator.tokenizer)    
        return text, None, None, logprobs

class SingleRAG(BasicRAG):
    def __init__(self, args, templates = None):
        super().__init__(args, templates = None)
        self.device = torch.device("cuda:1")
        self.generator.model.to(self.device)
    def inference(self, question, demo, case):

        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)

        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Below are the external knowledge references:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Please answer the question based on the external knowledge.\n"
        prompt += case

        print(prompt)
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, docs, [question]
    
    def inference_batch(self, question, choices):
        assert self.query_formulation == "direct"
        doc_list = []
        for query in question:    
            d = self.retrieve(query, topk = self.retrieve_topk)
            doc_list.append(d)

        prompt_list = []
        for idx in range(len(question)):
            prompt = ""
            prompt += "Below are the external knowledge references:\n"

            docs = doc_list[idx]
            for j, doc in enumerate(docs):
                prompt += f"[{j+1}] {doc}\n"
            prompt += "Please answer the question based on the external knowledge.\n"

            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            prompt = [prompt + self.inst, self.reply]
            final_shot = SHOT_TEMPLATES["mcq"].format(question=q, choices=c)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            
            prompt_list.append(prompt)

        text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        if self.use_counter == True:
            for i in range(len(question)):
                self.counter.add_generate(text[i], self.generator.tokenizer)
            
        return text, doc_list, question, logprobs

class OurRAG(BasicRAG):
    def __init__(self, args, examplars = None, templates = None):
        super().__init__(args)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.rm = self.retriever.model
        
        self.choice_num = 2

        self.device = torch.device("cuda:1")
        #self.device = torch.device("cuda:0")
        
        self.rm.to(device)        
        self.generator.model.to(self.device)
        
        self.generator.model.generation_config.temperature=None

        self.templates = templates
        if templates != None:
            self.inst = templates[0]
            self.reply = templates[1]
            self.answer = templates[2]
            self.knowledge_inst = templates[3]
            self.knowledge_inst_external = templates[4]

            self.knowledge_selection = templates[5]
            self.knowledge_refinement = ""

    def find_similar_k(self, document, query, topk = 3):
        #K, S
        doc_batch_tokens = self.tokenizer(document, padding=False, truncation=True)   
        #N, S'
        query_batch_tokens = self.tokenizer(query, padding = False, truncation = True)
        
        # Add padding
        doc_batch_tokens = self.tokenizer.pad(doc_batch_tokens, padding=True, return_tensors="pt")
        doc_batch_tokens = {k: v.cuda() for k,v in doc_batch_tokens.items()}
        
        query_batch_tokens = self.tokenizer.pad(query_batch_tokens, padding=True, return_tensors = "pt")
        query_batch_tokens = {k:v.cuda() for k,v in query_batch_tokens.items()}

        with torch.no_grad():
            #K x dim
            d_reps = self.rm(**doc_batch_tokens).pooler_output
            #1 x dim
            q_reps = self.rm(**query_batch_tokens).pooler_output
        d_reps.detach() 
        q_reps.detach()
        
        d_reps = torch.nn.functional.normalize(d_reps, p=2, dim=1)
        q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        #d_reps = d_reps.cpu().numpy()
        
        #N x K
        scores = torch.matmul(q_reps, d_reps.T)
        v, indices = torch.topk(scores, dim = -1, k = topk)
        #print(scores.shape, indices)
        
        ret_doc = []
        for i in range(len(query)):
            temp = []
            ret_temp = []
            for j in range(len(document)):
                if j in indices[i]:
                    temp.append(document[j])
                else:
                    ret_temp.append(document[j])

            ret_doc.append(temp)
            document = ret_temp
        return ret_doc

    def get_scores(self, document, query, topk = 3, normalize = True):
        #K, S
        doc_batch_tokens = self.tokenizer(document, padding=False, truncation=True)   
        #N, S'
        query_batch_tokens = self.tokenizer(query, padding = False, truncation = True)
         
        # Add padding
        doc_batch_tokens = self.tokenizer.pad(doc_batch_tokens, padding=True, return_tensors="pt")
        doc_batch_tokens = {k: v.cuda() for k,v in doc_batch_tokens.items()}
        
        query_batch_tokens = self.tokenizer.pad(query_batch_tokens, padding=True, return_tensors = "pt")
        query_batch_tokens = {k:v.cuda() for k,v in query_batch_tokens.items()}


        with torch.no_grad():
            #K x dim
            d_reps = self.rm(**doc_batch_tokens).pooler_output
            #1 x dim
            q_reps = self.rm(**query_batch_tokens).pooler_output
        d_reps.detach() 
        q_reps.detach()
        
        if normalize:        
            d_reps = torch.nn.functional.normalize(d_reps, p=2, dim=1)
            q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        #d_reps = d_reps.cpu().numpy()
        #N x K
        scores = torch.matmul(q_reps, d_reps.T)

        return scores

    def get_embedding(self, document, num_cluster = 3):
        #doc_batch_tokens = self.tokenizer(document, padding=False, truncation=True)   
        doc_batch_tokens = self.tokenizer(document, padding=False, truncation=True)   
        
        # Add padding
        doc_batch_tokens = self.tokenizer.pad(doc_batch_tokens, padding=True, return_tensors="pt")
        doc_batch_tokens = {k: v.cuda() for k,v in doc_batch_tokens.items()}

        with torch.no_grad():
            d_reps = self.rm(**doc_batch_tokens).pooler_output
        
        d_reps.detach() 
        d_reps = d_reps.cpu().numpy()
        #d_reps = torch.nn.functional.normalize(d_reps, p=2, dim=1)

        #cluster_ids_x, cluster_centers = kmeans(
        #    X=d_reps, num_clusters=2, distance='euclidean', device=torch.device('cuda')
        #)
        kmeans = KMeans(n_clusters=num_cluster, random_state=1353)
        kmeans.fit(d_reps)
        cluster_ids_x = kmeans.labels_
        
        #print(cluster_ids_x)

        docs = []
        for i in range(num_cluster):
            temp = []
            for idx, d in zip(cluster_ids_x, document):
                if idx == i:
                    temp.append(d)
            docs.append(temp)

        return docs

    def get_connection(self, document, query = None, num_docs = 4, deterministic = False):
        #doc_batch_tokens = self.tokenizer(document, padding=False, truncation=True)   
        doc_batch_tokens = self.tokenizer(document, padding=False, truncation=True)   
        
        tau = 1.0
        
        # Add padding
        doc_batch_tokens = self.tokenizer.pad(doc_batch_tokens, padding=True, return_tensors="pt")
        doc_batch_tokens = {k: v.cuda() for k,v in doc_batch_tokens.items()}

        with torch.no_grad():
            d_reps = self.rm(**doc_batch_tokens).pooler_output

        d_reps.detach() 
        #d_reps = torch.nn.functional.normalize(d_reps, p=2, dim=1)
        #d_reps = d_reps.cpu().numpy()
        
        if query != None:
            #N, S'
            q2d_score = self.get_scores(document, [query], normalize=False)[0]
            q2d_score.detach()
            #q2d_score = torch.cat([q2d_score[:rand_idx], q2d_score[rand_idx+1:]], dim = 0)
            scores = torch.softmax(q2d_score / tau, dim = -1)
            
            #print(scores)
            m = torch.distributions.categorical.Categorical(probs = scores)
            #rand_idx = m.sample()
            rand_idx = random.randint(0, len(document) - 1)
            q2d_score = torch.cat([q2d_score[:rand_idx], q2d_score[rand_idx+1:]], dim = 0)
        
        else:    
            rand_idx = random.randint(0, len(document) - 1)

        docs = [document[rand_idx]]
        #embedding = d_reps[rand_idx]
        #total_embedding = [d_reps[rand_idx]]
        embedding = d_reps[rand_idx]
        d_reps = torch.cat([d_reps[:rand_idx], d_reps[rand_idx+1:]], dim = 0)
        document = document[:rand_idx] + document[rand_idx+1:]

        #if query != None:
        #    #N, S'
        #    q2d_score = self.get_scores(document, [query], normalize=False)[0]
        #    q2d_score.detach()
        #    #q2d_score = torch.cat([q2d_score[:rand_idx], q2d_score[rand_idx+1:]], dim = 0)
            
        #print(q2d_score)
        
        for i in range(num_docs):
            #1, d
            #embedding = torch.tensor(total_embedding).mean(dim = 0)
            scores = torch.matmul(embedding / (i + 1), d_reps.T)
            #scores = torch.cosine_similarity(embedding / (i + 1), d_reps, dim = 1)
            
            if query != None:
                #print(scores, q2d_score)
                scores = scores + q2d_score
            #idx = torch.topk(scores, k = 1, dim = -1, largest = True)[1].item()
            scores = torch.softmax(scores / tau, dim = -1)
            m = torch.distributions.categorical.Categorical(probs = scores)
            #print(m.probs)
                
            if deterministic:
                idx = torch.topk(scores, k = 1, dim = -1, largest = True)[1].item()
            else:            
                idx = m.sample()
            
            docs.append(document[idx])
            embedding += d_reps[idx]
            d_reps = torch.cat([d_reps[:idx], d_reps[idx+1:]], dim = 0)
            document = document[:idx] + document[idx+1:]

            if query != None:
                q2d_score = torch.cat([q2d_score[:idx], q2d_score[idx+1:]], dim = 0)
            
        return docs

    def process_knowledge(self, generated_text, max_generated_knowledge = 3):
        generated_text = generated_text.strip()
        knowledge = generated_text.split("\n\n")[0].split("\n")
        #print(generated_text.split("\n\n"))
        
        knowledge = [k.replace("*", "") for k in knowledge]
        knowledge = [k.strip() for k in knowledge]
        knowledge = [k for k in knowledge if k]
        knowledge = knowledge[:max_generated_knowledge]
        #knowledge = [k for k in knowledge if k[-1] == "."]

        return knowledge

    def tokenizer_handler(
            self,
            inputs
        ):

        inputs = inputs[:, :-1]
        #inputs = inputs
        #print(inputs[:, -3:])

        return inputs

    def generate_text(
        self,
        prompt_list,
        do_sample = False,
        top_p = 1.0,
        max_new_tokens = 256):
        input_list = []
        
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]            
            messages = []
            for turn_id, turn in enumerate(prompt):
                if turn_id % 2 == 0:
                    messages.append({"role": "user", "content": turn})
                else:
                    messages.append({"role": "assistant", "content": turn})

            inputs = self.generator.tokenizer.apply_chat_template(messages, return_tensors="pt")#.to(device)
            inputs = self.tokenizer_handler(inputs)
        
            input_list.append(inputs[0])
        
        
        max_length = [len(s) for s in input_list]
        max_length = max(max_length)
        
        #print(max_length)
        
        input_ids = torch.zeros((len(input_list), max_length)).to(torch.long).fill_(self.generator.tokenizer.pad_token_id)
        attention_mask = torch.zeros((len(input_list), max_length)).to(torch.long)
        
        for idx in range(len(input_list)):
            length = len(input_list[idx])
            #print(input_list[idx])
            
            input_ids[idx, -length:] = input_list[idx]
            attention_mask[idx, -length:] = 1
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.generator.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens, 
            do_sample=do_sample,
            top_p= top_p,
            num_beams=1,
            return_dict_in_generate = True, 
            output_scores = True,
        )

        input_length = input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        text = self.generator.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
        tokens = []
        for sentence in generated_tokens:
            tokens += [self.generator.tokenizer.decode(t, skip_special_tokens = True) for t in sentence]

        logprobs = torch.softmax(outputs.scores[-1], dim=-1)#[0]
        logprobs = logprobs.cpu().numpy()

        return text, tokens, logprobs

    def retrieve_from_knowledge_list(self, question, knowledge_text_list, choices = None,  query_expansion = True, zebra = False, topk = 3):
        q_list = []
        doc_list = []
        score_list = []
        for idx, knowledge_text in enumerate(knowledge_text_list):

            if query_expansion:
                if isinstance(knowledge_text, str):
                    queries = [knowledge_text]
                else:
                    queries = knowledge_text
                
                if choices != None:        
                    question_prompt = question[idx] 
                    choice_prompt = [f"[SEP] {c_['label']}. {c_['text']}".strip() for c_ in choices[idx]]
                    choice_prompt = " ".join(choice_prompt)
                    question_prompt += choice_prompt
                else:
                    question_prompt = question[idx]
                
                if queries == []:
                    queries = [question_prompt]
                    
                else:
                    queries.append(question_prompt)

            else:
                queries = [knowledge_text]

            docs = []
            scores = []
            for query in queries:
                d, score = self.retrieve(query, topk = topk)
                if zebra:
                    temp = []
                    for document in d:
                        q, c, knowledge = document
                        c = [f"* {c_['text']}" for c_ in c]
                        c = "\n".join(c)
                        
                        knowledge = [k if k.endswith(".") else f"{k}." for k in knowledge]
                        knowledge = [f"* {k}" for k in knowledge]
                        knowledge = "\n".join(knowledge)
                        knowledge = f"Explanations:\n{knowledge}"
                        temp.append(knowledge)
                    
                    d = temp
                    docs += d
                else:                            
                    docs += d
                    scores += score

            doc_list.append(docs)
            score_list.append(scores)
            
            q_list.append(queries)
        
        return q_list, doc_list, score_list

    def document_prompts(self, docs, number = True, is_explanation = False):
        if is_explanation:
            temp = "Belows are the list of explanations:\n"
        else:
            temp = "\nBelows are the external knowledge references:\n"

        num = 0
        for j, d in enumerate(docs):
            if number:
                temp += f"[{num + 1}] {d}\n"
            else:
                temp += f"* {d}\n"
            num += 1

        temp += "\n"

        return temp


    def generate_prompts(self, instruction, question, choice, template_type, document = None, explanation = None):
        if document != None:
            instruction = document + "\n\n" + instruction

        reply = self.reply
        prompt = [instruction, reply]
        choice = [f"* {c_['label']}: {c_['text']}".strip() for c_ in choice]
        choice = "\n".join(choice)

        if explanation != None:
            shot = SHOT_TEMPLATES[template_type].format(question=question, choices=choice, knowledge = explanation)
        else:
            shot = SHOT_TEMPLATES[template_type].format(question=question, choices=choice)

        if template_type == "mcq_with_kg":
            assistant_reply = "Answer: "
        elif template_type == "knowledge_selection":
            assistant_reply = "Useful External Knowledge:\n* "
        elif template_type == "knowledge_refinement":
            assistant_reply = "Refined Explanations:\n* "
        elif template_type == "knowledge_aggregation":
            assistant_reply = "Aggegated Explanation:\n* "
        else:
            assistant_reply = "Explanations:\n* "

        prompt += [shot, assistant_reply]
        return prompt

    def inference_batch(self, question, choices, query_expansion = True, knowledge_generation = True, knowledge_aggregation = True, N = 3):        
        if query_expansion:
            knowledge_prompt = []
            for i in range(len(question)):
                prompt = self.generate_prompts(self.knowledge_inst, question[i], choices[i], "mcq")
                knowledge_prompt.append(prompt)

            #Generate the explanations for retrieval
            old_knowledge_text_list, _, _ = self.generate_text(knowledge_prompt, max_new_tokens = 196)
            knowledge_text_list = [self.process_knowledge(k, self.choice_num) for k in old_knowledge_text_list]
            retrieve_topk = self.retrieve_topk
        else:
            knowledge_text_list = question
            retrieve_topk = self.retrieve_topk * (self.choice_num + 1)
            
        log_prob_list = []
        
        if knowledge_generation:
            doc_list = []
            explanations = knowledge_text_list
            selected_explanations_list = []
            selected_choices = None
            
            #N', K
            q_list, selected_doc_list, document_scores = self.retrieve_from_knowledge_list(question, explanations, selected_choices, query_expansion, zebra = False, topk = retrieve_topk)
            sub_prompt_list = []
            
            for selected_num in range(len(question)):
                d = selected_doc_list[selected_num]
                temp = []
                score_temp = []        
                for j in range(len(d)):
                    if d[j] not in temp:
                        temp.append(d[j])
                        score_temp.append(document_scores[selected_num][j])
                selected_doc_list[selected_num] = temp
                document_scores[selected_num] = score_temp

                question_prompt = question[selected_num] 

                if len(selected_doc_list[selected_num]) > self.retrieve_topk:
                    document_num = self.retrieve_topk
                else:
                    document_num = len(selected_doc_list[selected_num]) - 1
                
                document_list = []
                for j in range(N):
                    if self.sampling == 'cluster':          
                        d_cluster = self.get_embedding(selected_doc_list[selected_num], num_cluster = document_num + 1)
                        d_temp = []
                        for k in range(document_num + 1):
                            if len(d_cluster[k]) < 1:
                                continue                            
                            d_temp.append(random.sample(d_cluster[k],1)[0])

                    elif self.sampling == "topk":
                        d_temp = []
                        top_indices = np.argsort(document_scores[selected_num], axis = 0)
                        top_indices = top_indices[-(document_num + 1):]
                        for k in top_indices:
                            d_temp.append(selected_doc_list[selected_num][k])
                            
                    elif self.sampling == 'random':
                        d_temp = random.sample(selected_doc_list[selected_num], document_num + 1)
                        
                    elif self.sampling == "deterministic":
                        d_temp = self.get_connection(selected_doc_list[selected_num], num_docs=document_num, query = question_prompt, deterministic=True)

                    elif self.sampling == "reconnect": 
                        if len(selected_doc_list[selected_num]) == 1:
                            print(selected_doc_list[selected_num])
                            print(q_list)
                            d_temp = selected_doc_list[selected_num]
                        else:
                            d_temp = self.get_connection(selected_doc_list[selected_num], num_docs=document_num, query = question_prompt)

                    document_list.append(d_temp)
                    temp = self.document_prompts(d_temp, number = False)
                    prompt = self.generate_prompts(self.knowledge_inst_external, question[selected_num], choices[selected_num], "knowledge_generation", document = temp)
                    sub_prompt_list.append(prompt)

                doc_list.append(document_list)

            #(B X N, )
            if self.sampling == "topk":
                selected_sub_explanations, _, _ = self.generate_text(sub_prompt_list, max_new_tokens = 196, top_p = 0.7, do_sample= True)
            else:
                selected_sub_explanations, _, _ = self.generate_text(sub_prompt_list, max_new_tokens = 196)
            
            temp = []
            for i in range(len(selected_sub_explanations) // N):
                temp.append(selected_sub_explanations[i * N : (i + 1) * N])

            selected_explanations_list = temp
            
            if knowledge_aggregation:
                prompt_list = []
                for selected_num in range(len(question)):        
                    new = []
                    for number, s in enumerate(selected_explanations_list[selected_num]):
                        s = s.strip()
                        s = s.split("\n\n")[0].replace("*","")
                        s = f"* " + s
                        new.append(s)
                    new = "\n".join(new)
                    new += "\n\n"
                    new = "Belows are the list of explanations:\n" + new
                    sub_explanations = new
                    
                    prompt = self.generate_prompts(self.knowledge_selection, question[selected_num], choices[selected_num], "knowledge_aggregation", document = sub_explanations)
                    prompt_list.append(prompt)

                generated_knowledge, _, _ = self.generate_text(prompt_list, max_new_tokens = 196)

            else:
                generated_knowledge = selected_explanations_list


        elif knowledge_aggregation:
            q_list, doc_list = self.retrieve_from_knowledge_list(question, knowledge_text_list, None, query_expansion, zebra = False)
            prompt_list = []
            for selected_num in range(len(question)):        
                new = self.document_prompts(doc_list[selected_num], number = False)
                prompt = self.generate_prompts(self.knowledge_selection, question[selected_num], choices[selected_num], "knowledge_aggregation", document = new)
                prompt_list.append(prompt)

            #(B, )
            generated_knowledge, _, _ = self.generate_text(prompt_list, max_new_tokens = 196)
            explanations = doc_list
            selected_explanations_list = doc_list
        else:
            
            q_list, doc_list = self.retrieve_from_knowledge_list(question, knowledge_text_list, None, query_expansion, zebra = False)
            generated_knowledge = doc_list
            explanations = doc_list
            selected_explanations_list = doc_list

        prompt_list = []
        for idx in range(len(question)):   
            docs = doc_list[idx]
            temp = self.document_prompts(docs, number = False)
            
            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            if knowledge_generation or knowledge_aggregation:
                prompt = [self.inst, self.reply]
                if knowledge_aggregation:            
                    if self.dataset in ["wg", "wsc"]:
                        generated_knowledge[idx] = generated_knowledge[idx]
                    else:
                        generated_knowledge[idx] = generated_knowledge[idx].strip().split('\n\n')[0]#.replace("*","")

                    sample_knowledge = [generated_knowledge[idx]]
                else:
                    sample_knowledge = generated_knowledge[idx]
                sample_knowledge = [k if k.endswith(".") else f"{k}." for k in sample_knowledge]
                sample_knowledge = [f"* {k.strip()}" for k in sample_knowledge]
                sample_knowledge = "\n".join(sample_knowledge)
                final_shot = SHOT_TEMPLATES["mcq_with_kg"].format(question=q, choices=c, knowledge = sample_knowledge)
                
            else:
                prompt = [self.inst, self.reply]
                sample_knowledge = generated_knowledge[idx]
                sample_knowledge = [k if k.endswith(".") else f"{k}." for k in sample_knowledge]
                sample_knowledge = [f"* {k.strip()}" for k in sample_knowledge]
                sample_knowledge = "\n".join(sample_knowledge)
                
                final_shot = SHOT_TEMPLATES["mcq_with_kg"].format(question=q, choices=c, knowledge = sample_knowledge)
            #print(final_shot)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            prompt_list.append(prompt)
            
            print("=====================================================")
            print("".join(prompt))
            print("=====================================================")

        generated_answer, _, log_prob_list = self.generate_text(prompt_list, max_new_tokens = 1)
        return generated_answer, doc_list, explanations, np.array(log_prob_list), generated_knowledge, selected_explanations_list


class ZEBRA(BasicRAG):
    def __init__(self, args, examplars = None, templates = None):
        super().__init__(args)

        self.args = args

        self.device = torch.device("cuda:0")
        self.generator.model.to(self.device)
        if examplars == None:
            examplars = ["","",""]            

        #from zebra.prompts import SHOT_TEMPLATES
        self.templates = templates
        if templates != None:
            self.inst = templates[0]
            self.reply = templates[1]
            self.answer = templates[2]
            self.knowledge_inst = templates[3]

    def process_knowledge(self, generated_text, max_generated_knowledge = 3):
        knowledge = generated_text.split("\n\n")[0].split("\n")
        knowledge = [k.replace("*", "") for k in knowledge]
        knowledge = [k.strip() for k in knowledge]
        knowledge = [k for k in knowledge if k]
        knowledge = knowledge[:max_generated_knowledge]
        
        return knowledge

    def tokenizer_handler(
            self,
            inputs
        ):
        """
        Handle the tokenizer inputs.

        Parameters:
        - inputs (torch.Tensor): Inputs.
        - tokenizer (AutoTokenizer): Tokenizer.

        Returns:
        - torch.Tensor: Processed inputs.
        """
        inputs = inputs[:, :-1]
        return inputs

    def generate_text(
        self,
        prompt_list,
        max_new_tokens = 256):
        """
        Generate text using the model.

        Parameters:
        - model (AutoModelForCausalLM): Model.
        - tokenizer (AutoTokenizer): Tokenizer.
        - prompt (List[str]): List of conversation turns.
        - max_new_tokens (int): Maximum number of new tokens.
        - device (str): Device to run the model on (default is 'cuda' if available, otherwise 'cpu').

        Returns:
        - tuple: A tuple containing the model outputs and the generated text.
        """
        # Build the conversation turns in the format required by chat templates.
        
        input_list = []
        #attention_mask = []
        
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]            
            messages = []
            for turn_id, turn in enumerate(prompt):
                if turn_id % 2 == 0:
                    messages.append({"role": "user", "content": turn})
                else:
                    messages.append({"role": "assistant", "content": turn})

            inputs = self.generator.tokenizer.apply_chat_template(messages, return_tensors="pt")#.to(device)
            inputs = self.tokenizer_handler(inputs)
        
            input_list.append(inputs[0])
        
        
        max_length = [len(s) for s in input_list]
        max_length = max(max_length)
        
        input_ids = torch.zeros((len(input_list), max_length)).to(torch.long).fill_(self.generator.tokenizer.pad_token_id)
        attention_mask = torch.zeros((len(input_list), max_length)).to(torch.long)
        
        for idx in range(len(input_list)):
            length = len(input_list[idx])
            #print(input_list[idx])
            
            input_ids[idx, -length:] = input_list[idx]
            attention_mask[idx, -length:] = 1
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        #attention_mask = torch.ones_like(inputs).to(device)
        
        outputs = self.generator.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens, 
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            return_dict_in_generate = True, 
            output_scores = True,
        )

        input_length = input_ids.shape[1]

        generated_tokens = outputs.sequences[:, input_length:]

        text = self.generator.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
        #text = [self.tokenizer.decode(t) for t in generated_tokens]
        tokens = []
        for sentence in generated_tokens:
            tokens += [self.generator.tokenizer.decode(t, skip_special_tokens = True) for t in sentence]

        logprobs = torch.softmax(outputs.scores[-1], dim=-1)#[0]
        logprobs = logprobs.cpu().numpy()

        
        return text, tokens, logprobs

        #return model_outputs, generated_text
        
    def inference_batch(self, question, choices, k = 5):
        knowledge_text_list = question
        score_list = []
        prompt_list = []
        doc_list = []
        for idx, knowledge_text in enumerate(knowledge_text_list):
            queries = [knowledge_text]
            docs = []
            for query in queries:    
                d = self.retrieve(query, topk = k)
                docs.append(d)

            prompt = [self.knowledge_inst, self.reply]# + 
            shots = []
            for document in d:
                q, c, knowledge = document
                c = [f"* {c_['text']}" for c_ in c]
                c = "\n".join(c)
                
                knowledge = [k if k.endswith(".") else f"{k}." for k in knowledge]
                knowledge = [f"* {k}" for k in knowledge]
                knowledge = "\n".join(knowledge)
                knowledge = f"Explanations:\n{knowledge}"
                
                shot = SHOT_TEMPLATES["knowledge_generation"].format(question=q, choices=c)
                shots.append(shot)
                shots.append(knowledge)
            
            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['text']}" for c_ in c]
            c = "\n".join(c)

            final_shot = SHOT_TEMPLATES["knowledge_generation"].format(question=q, choices=c)
            assistant_reply = "Explanations:\n* "
            
            prompt += shots + [final_shot, assistant_reply]
            
            doc_list.append(docs)
            prompt_list.append(prompt)

        # Generate Explanations
        generated_knowledge, _, _ = self.generate_text(prompt_list)
        generated_knowledge = [self.process_knowledge(k, 3) for k in generated_knowledge]

        prompt_list = []
        for idx in range(len(generated_knowledge)):   
            prompt = [self.inst, self.reply]         
            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)

            sample_knowledge = generated_knowledge[idx]
            sample_knowledge = [k if k.endswith(".") else f"{k}." for k in sample_knowledge]
            sample_knowledge = [f"* {k.strip()}" for k in sample_knowledge]
            sample_knowledge = "\n".join(sample_knowledge)
            
            final_shot = SHOT_TEMPLATES["mcq_with_kg"].format(question=q, choices=c, knowledge = sample_knowledge)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            prompt_list.append(prompt)

        #print(prompt_list)
        generated_answer, _, log_prob_list = self.generate_text(prompt_list, max_new_tokens = 1)

        return generated_answer, doc_list, question, np.array(log_prob_list), generated_knowledge


from typing import List, Dict, Tuple
import os
import time
import tqdm
import uuid
import numpy as np
import torch
import faiss
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search.lexical.elastic_search import ElasticSearch

from model import GoldenRetrieverModel
from datasets import load_dataset
import json
import pickle

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def get_random_doc_id():
    return f'_{uuid.uuid4()}'

class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        index_name: str = None,
        engine: str = 'elasticsearch',
        **search_engine_kwargs,
    ):
        self.tokenizer = tokenizer
        # load index
        assert engine in {'elasticsearch', 'bing'}
        if engine == 'elasticsearch':
            self.max_ret_topk = 1000
            self.retriever = EvaluateRetrieval(
                BM25Search(index_name=index_name, hostname='localhost', initialize=False, number_of_shards=1),
                k_values=[self.max_ret_topk])

    def retrieve(
        self,
        queries: List[str],  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
    ):
        assert topk <= self.max_ret_topk
        device = None
        bs = len(queries)

        # truncate queries
        if max_query_length:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            # truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # retrieve
        results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), queries)), disable_tqdm=True)

        # prepare outputs
        docids: List[str] = []
        docs: List[str] = []
        for qid, query in enumerate(queries):
            _docids: List[str] = []
            _docs: List[str] = []
            if qid in results:
                for did, (score, text) in results[qid].items():
                    _docids.append(did)
                    _docs.append(text)
                    if len(_docids) >= topk:
                        break
            if len(_docids) < topk:  # add dummy docs
                _docids += [get_random_doc_id() for _ in range(topk - len(_docids))]
                _docs += [''] * (topk - len(_docs))
            docids.extend(_docids)
            docs.extend(_docs)

        docids = np.array(docids).reshape(bs, topk)  # (bs, topk)
        docs = np.array(docs).reshape(bs, topk)  # (bs, topk)
        return docids, docs


def bm25search_search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
    # Index the corpus within elastic-search
    # False, if the corpus has been already indexed
    if self.initialize:
        print("initialize")
        self.index(corpus)
        # Sleep for few seconds so that elastic-search indexes the docs properly
        time.sleep(self.sleep_for)

    #retrieve results from BM25
    query_ids = list(queries.keys())
    queries = [queries[qid] for qid in query_ids]

    final_results: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que', disable=kwargs.get('disable_tqdm', False)):
        query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
        results = self.es.lexical_multisearch(
            texts=queries[start_idx:start_idx+self.batch_size],
            top_hits=top_k)
        for (query_id, hit) in zip(query_ids_batch, results):
            scores = {}
            for corpus_id, score, text in hit['hits']:
                scores[corpus_id] = (score, text)
                final_results[query_id] = scores

    return final_results

BM25Search.search = bm25search_search


def elasticsearch_lexical_multisearch(self, texts: List[str], top_hits: int, skip: int = 0) -> Dict[str, object]:
    """Multiple Query search in Elasticsearch

    Args:
        texts (List[str]): Multiple query texts
        top_hits (int): top k hits to be retrieved
        skip (int, optional): top hits to be skipped. Defaults to 0.

    Returns:
        Dict[str, object]: Hit results
    """
    request = []

    assert skip + top_hits <= 10000, "Elastic-Search Window too large, Max-Size = 10000"

    for text in texts:
        req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
        req_body = {
            "_source": True, # No need to return source objects
            "query": {
                "multi_match": {
                    "query": text, # matching query with both text and title fields
                    "type": "best_fields",
                    "fields": [self.title_key, self.text_key],
                    "tie_breaker": 0.5
                    }
                },
            "size": skip + top_hits, # The same paragraph will occur in results
            }
        request.extend([req_head, req_body])

    res = self.es.msearch(body = request)

    result = []
    for resp in res["responses"]:
        responses = resp["hits"]["hits"][skip:] if 'hits' in resp else []

        hits = []
        for hit in responses:
            hits.append((hit["_id"], hit['_score'], hit['_source']['txt']))

        result.append(self.hit_template(es_res=resp, hits=hits))
    return result

ElasticSearch.lexical_multisearch = elasticsearch_lexical_multisearch


def elasticsearch_hit_template(self, es_res: Dict[str, object], hits: List[Tuple[str, float]]) -> Dict[str, object]:
    """Hit output results template

    Args:
        es_res (Dict[str, object]): Elasticsearch response
        hits (List[Tuple[str, float]]): Hits from Elasticsearch

    Returns:
        Dict[str, object]: Hit results
    """
    result = {
        'meta': {
            'total': es_res['hits']['total']['value'] if 'hits' in es_res else None,
            'took': es_res['took'] if 'took' in es_res else None,
            'num_hits': len(hits)
        },
        'hits': hits,
    }
    return result

ElasticSearch.hit_template = elasticsearch_hit_template


DATA_PATH = {
    "csqa" : "/home/user10/RALM_CSQA/data/zebra/csqa-train.jsonl",
    "csqa2" : "/home/user10/RALM_CSQA/data/zebra/csqa2-train.jsonl",
    "piqa" : "/home/user10/RALM_CSQA/data/zebra/piqa-train.jsonl"
}


        
#  Successfully uninstalled datasets-2.14.1 예전 버전 - dragin
class DPR_ZEBRA:
    def __init__(
        self, 
        model_name_or_path,
        sgpt_encode_file_path,
        passage_file,
        augmentation = False,
        datasets = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = GoldenRetrieverModel.from_pretrained("sapienzanlp/zebra-retriever-e5-base-v2")
        print(self.model)
        
        self.model.cuda(0)
        self.model.eval()

        print("Building DPR indexes")

        self.p_reps = []

        encode_file_path = sgpt_encode_file_path
        dir_names = sorted(os.listdir(encode_file_path))
        
        
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.IndexFlatL2(768)
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)
        self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        
        question_to_explanations = dict()
        explanations = load_dataset("sapienzanlp/zebra-kb-explanations", "all")["train"]
        for sample in explanations:
            question_id = sample["id"]
            question_to_explanations[question_id] = sample["positives"]
        
        with open("../data/documents.jsonl", "r") as f:
            x = f.readlines()
        
        print(len(question_to_explanations), len(x))
        
        if datasets != None:
            with open(DATA_PATH[datasets], "r") as f:
                train_data = f.readlines()
             
            id_dict = dict()   
            #id_dict = {s["id"]: len(id_dict) for s in train_data}
            for s in train_data:
                s = json.loads(s[:-1])
                id_dict[s["id"]] = len(id_dict)
        
        indices = []

        self.docs = []
        skip_num = 0
        for idx, s in enumerate(x):
            s = json.loads(s[:-1])
            question_id = s["id"]
            
            if datasets != None:
                if question_id in id_dict:
                    indices.append(idx)
                    skip_num += 1
                    continue
            
            if question_id not in question_to_explanations:
                skip_num += 1
                #print(question_id)
                s["explanation"] = ""
            else:
                s["explanation"] = question_to_explanations[question_id]
                
            #print(s)
            explanations = s["explanation"]
            example_question = s["text"].split(" [SEP] ")[0]
            example_choices = s["text"].split(" [SEP] ")[1:]
            example_choices = [
                {
                    "label": choice.split(".")[0].strip(),
                    "text": choice.split(".")[1].strip(),
                }
                for choice in example_choices
            ]

            self.docs.append([example_question, example_choices, explanations])
            
        print(len(self.docs), len(question_to_explanations), skip_num)
        
        embeddings = torch.load("../data/embeddings.pt", map_location="cpu")
        print("Embeddings : ", embeddings.shape)
        embeddings = embeddings.cpu().tolist()
        
        temp = []
        for idx in range(len(embeddings)):
            if idx in indices:
                continue
            temp.append(embeddings[idx][:])
        temp = np.array(temp).astype(np.float32)
        embeddings = temp

        print(len(self.docs), "Embeddings : ", embeddings.shape)
        self.faiss_index.train(embeddings)  
        self.faiss_index.add(embeddings)
        print(self.faiss_index.ntotal)
        
    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        batch_tokens = {k: v.cuda() for k,v in batch_tokens.items()}
        return batch_tokens

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        #print(queries)
        batch_tokens = self.tokenize_with_specb(queries, is_query=True)
        
        with torch.no_grad():
            q_reps = self.model(**batch_tokens).pooler_output
        
        q_reps.detach()
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        topk_values_list = []
        topk_indices_list = []
        
        topk_indices_list_custom = []
        
        Distance, Index = self.faiss_index.search(q_reps.cpu().numpy(), k = topk)

        psgs = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                psg = self.docs[Index[qid][j]]
                ret.append(psg)
            psgs.append(ret)
        return psgs

class DPR:
    def __init__(
        self, 
        retrieval_model_name_or_path = "./retriever/question_encoder",
        embedding_path = "./documents/",
        passage_file = None
    ):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = GoldenRetrieverModel.from_pretrained(retrieval_model_name_or_path)
        self.embedding_path = embedding_path

        self.device = torch.device("cuda:0")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Building DPR indexes")

        self.p_reps = []

        res = faiss.StandardGpuResources()
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)
        print(self.embedding_path)
        self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        self.faiss_index2 = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index2 = faiss.index_cpu_to_gpu(res, 1, self.faiss_index2)

        #split_parts = 3
        split_parts = 4
        
        self.docs = []
        self.docs2 = []
        for i in tqdm.tqdm(range(split_parts), ncols =100):
            st = time.time()

            f = open(self.embedding_path + "/explanation_embeddings%d/documents.jsonl"%(i+1), "r")

            docs = f.readlines()
            print(json.loads(docs[0]))
            docs = [json.loads(s)["text"] for s in docs]
            
            tp = torch.load(self.embedding_path + "/explanation_embeddings%d/embeddings.pt"%(i+1), map_location="cpu")    
            
            print("Embeddings : ", tp.shape)
            
            tp = torch.nn.functional.normalize(tp, p=2, dim=1)
            tp = tp.numpy().astype(np.float32)
            print("Spended_time:" , time.time() - st)
            
            if i > 1:
                self.faiss_index2.train(tp)
                self.faiss_index2.add(tp)
                self.docs2 += docs
            else:
                self.faiss_index.train(tp)            
                self.faiss_index.add(tp)
                self.docs += docs
            
            print(self.faiss_index.ntotal, self.faiss_index2.ntotal)
            
            ft = time.time()
            print("Spended_time:" , ft - st)

        print(len(self.docs), len(self.docs2))
        self.doc_length = len(self.docs)
        self.docs += self.docs2
        print(len(self.docs))

    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        batch_tokens = {k: v.to(self.device) for k,v in batch_tokens.items()}
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        # Get the embeddings
        with torch.no_grad():
            last_hidden_state, _, _ = self.model(**batch_tokens)

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        #print(queries)
        batch_tokens = self.tokenize_with_specb(queries, is_query=True)

        with torch.no_grad():
            q_reps = self.model(**batch_tokens).pooler_output

        q_reps.detach()
        q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=1)

        Distance, Index = self.faiss_index.search(q_reps.cpu().numpy(), k = topk)
        Distance2, Index2 = self.faiss_index2.search(q_reps.cpu().numpy(), k = topk)

        Index2 = Index2 + self.doc_length
        
        SuperIndex = np.concatenate([Index,Index2], axis = 1)
        SuperDistance = np.concatenate([Distance, Distance2], axis = 1)
        top_indices = np.argsort(SuperDistance, axis = 1)
        top_indices = top_indices[0][-topk:]
        Index = [SuperIndex[0][top_indices]]

        psgs = []
        distances = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                #idx = global_topk_indices[j][qid].item()
                #fid, rk = idx // topk, idx % topk
                psg = self.docs[Index[qid][j]]
                distances.append(SuperDistance[qid][j])
                #print(psg)
                ret.append(psg)
            psgs.append(ret)
        return psgs, distances

class DPR_Single:
    def __init__(
        self, 
        retrieval_model_name_or_path = "/home/user10/RALM_CSQA/retriever/question_encoder",
        embedding_path = "/home/user10/RALM_CSQA/documents/",
        passage_file = None
    ):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = GoldenRetrieverModel.from_pretrained(retrieval_model_name_or_path)
        self.embedding_path = embedding_path

        self.device = torch.device("cuda:0")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Building DPR indexes")

        self.p_reps = []

        res = faiss.StandardGpuResources()
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)
        print(self.embedding_path)
        self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        #self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        #quantizer = faiss.IndexFlatL2(768)        
        #self.faiss_index = faiss.IndexIVFFlat(self.faiss_index, 768, 66536)
        #self.faiss_index = faiss.IndexIVFPQ(quantizer, 768, 1024, 8, faiss.METRIC_L2)
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        #split_parts = 3
        split_parts = 4
        
        self.docs = []
        self.docs2 = []
        for i in tqdm.tqdm(range(split_parts), ncols =100):
            st = time.time()

            f = open(self.embedding_path + "/explanation_embeddings%d/documents.jsonl"%(i+1), "r")

            docs = f.readlines()
            print(json.loads(docs[0]))
            docs = [json.loads(s)["text"] for s in docs]
            
            tp = torch.load(self.embedding_path + "/explanation_embeddings%d/embeddings.pt"%(i+1), map_location="cpu")    
            
            print("Embeddings : ", tp.shape)
            
            tp = torch.nn.functional.normalize(tp, p=2, dim=1)
            tp = tp.numpy().astype(np.float32)
            print("Spended_time:" , time.time() - st)

            self.faiss_index.train(tp)            
            self.faiss_index.add(tp)
            self.docs += docs
            
            print(self.faiss_index.ntotal)
            
            ft = time.time()
            print("Spended_time:" , ft - st)

        print(len(self.docs))
        self.doc_length = len(self.docs)
        print(len(self.docs))

    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        batch_tokens = {k: v.to(self.device) for k,v in batch_tokens.items()}
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            
            #last_hidden_state = self.model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state
            last_hidden_state, _, _ = self.model(**batch_tokens)


        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        #print(queries)
        batch_tokens = self.tokenize_with_specb(queries, is_query=True)
        
        #print(batch_tokens)
        
        with torch.no_grad():
            q_reps = self.model(**batch_tokens).pooler_output
        
        
        #print(q_reps)

        q_reps.detach()
        q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=1)

        Distance, Index = self.faiss_index.search(q_reps.cpu().numpy(), k = topk)

        psgs = []
        distances = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                #idx = global_topk_indices[j][qid].item()
                #fid, rk = idx // topk, idx % topk
                psg = self.docs[Index[qid][j]]
                distances.append(Distance[qid][j])
                #print(psg)
                ret.append(psg)
            psgs.append(ret)
        return psgs


class DPR_Test:
    def __init__(
        self, 
        retrieval_model_name_or_path = "../retriever/question_encoder",
        embedding_path = "../documents/",
        passage_file = None
    ):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = GoldenRetrieverModel.from_pretrained(retrieval_model_name_or_path)
        self.embedding_path = embedding_path

        self.device = torch.device("cuda:0")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Building DPR indexes")

        self.p_reps = []

        #encode_file_path = sgpt_encode_file_path
        #dir_names = sorted(os.listdir(encode_file_path))

        res = faiss.StandardGpuResources()
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)
        print(self.embedding_path)
        self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        self.docs = []
        st = time.time()
        f = open(self.embedding_path + "/explanation_embeddings/documents.jsonl", "r")
        docs = f.readlines()
        print(json.loads(docs[0]))
        docs = [json.loads(s)["text"] for s in docs]
        tp = torch.load(self.embedding_path + "/explanation_embeddings/embeddings.pt", map_location="cpu")    
        print("Embeddings : ", tp.shape)
        
        tp = torch.nn.functional.normalize(tp, p=2, dim=1)
        tp = tp.numpy().astype(np.float32)
        print("Spended_time:" , time.time() - st)
        self.faiss_index.train(tp)            
        self.faiss_index.add(tp)
        self.docs += docs
        print(self.faiss_index.ntotal)
        
        ft = time.time()
        print("Spended_time:" , ft - st)

        print(len(self.docs))

    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        batch_tokens = {k: v.to(self.device) for k,v in batch_tokens.items()}
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state, _, _ = self.model(**batch_tokens)


        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        #print(queries)
        batch_tokens = self.tokenize_with_specb(queries, is_query=True)
        
        #print(batch_tokens)
        
        with torch.no_grad():
            q_reps = self.model(**batch_tokens).pooler_output
        
        
        #print(q_reps)

        q_reps.detach()
        q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        Distance, Index = self.faiss_index.search(q_reps.cpu().numpy(), k = topk)

        psgs = []
        distances = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                #idx = global_topk_indices[j][qid].item()
                #fid, rk = idx // topk, idx % topk
                psg = self.docs[Index[qid][j]]
                distances.append(Distance[qid][j])
                #print(psg)
                ret.append(psg)
            psgs.append(ret)
        return psgs, distances


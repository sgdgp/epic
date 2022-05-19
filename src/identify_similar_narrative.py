import json
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AlbertModel, AlbertTokenizer
from transformers import BartTokenizer, BartModel
from transformers import T5Model, T5Tokenizer, T5EncoderModel
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import random
from random import shuffle
import os
import sys
from tqdm import trange, tqdm
from scipy.spatial.distance import jensenshannon
import argparse
import logging

hyper_params = {
    "seed" : 42,
    "max_length_quote": 33,
    "max_length_narrative": 256,
    "batch_size": 16,
    "num_epochs": 15,
    "learning_rate": 1e-5,
    "train_test_split": 0.6,
    "dataset_file": "./data/full_dataset.json",
    "experiment_name" : "similar_narratives" 
}



random.seed(hyper_params["seed"])
np.random.seed(hyper_params["seed"])
torch.manual_seed(hyper_params["seed"])
torch.cuda.manual_seed(hyper_params["seed"])

device = torch.device("cuda")

CACHE_DIR = "./hf_cache/"



LLM_model = {
    "bert-base" : BertModel.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR, add_pooling_layer=True),
    "roberta-base":RobertaModel.from_pretrained("roberta-base", cache_dir=CACHE_DIR, add_pooling_layer=True),
    "distilbert-base-uncased" : DistilBertModel.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_DIR),
    # "albert-base-v2" : AlbertModel.from_pretrained('albert-base-v2', cache_dir=CACHE_DIR),
    # "bart-base" : BartModel.from_pretrained("facebook/bart-base", cache_dir=CACHE_DIR), 
    # "t5-base" : T5EncoderModel.from_pretrained("t5-base", cache_dir=CACHE_DIR),
    # "gpt2-base" : GPT2Model.from_pretrained("gpt2", cache_dir=CACHE_DIR),
    "sentencebert" : AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR)
}

LLM_tokenizer = {
    "bert-base" : BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR),
    "roberta-base" : RobertaTokenizer.from_pretrained("roberta-base", cache_dir=CACHE_DIR),
    "distilbert-base-uncased" : DistilBertTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_DIR),
    # "albert-base-v2" : AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=CACHE_DIR),
    # "bart-base" : BartTokenizer.from_pretrained("facebook/bart-base", cache_dir=CACHE_DIR),
    # "t5-base" : T5Tokenizer.from_pretrained("t5-base", cache_dir=CACHE_DIR),
    # "gpt2-base" : GPT2Tokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR),
    "sentencebert" : AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR)
}

class Model(nn.Module):
    def __init__(self, quote_tensors, model_type):
        super(Model, self).__init__()
        self.bert_model_type = model_type
        self.bert_model = LLM_model[self.bert_model_type]
        # BertModel.from_pretrained('bert-base-uncased', cache_dir = "./")
        
        self.bert_model = self.bert_model.to(device)
        self.quote_tensors = quote_tensors

    def set_train(self):
        self.bert_model.train()
        self.train()

    def set_eval(self):
        self.bert_model.eval()
        self.eval()

    def forward(self, batch_narrative_tensor):
        if self.bert_model_type == "bert-base" or self.bert_model_type == "albert-base-v2":
            
            cls_q = self.bert_model(**self.quote_tensors).last_hidden_state
            cls_n = self.bert_model(**batch_narrative_tensor).last_hidden_state

            cls_q = cls_q[: , 0, :]
            cls_n = cls_n[: , 0, :]
        
        elif self.bert_model_type == "roberta-base":
            cls_q = self.bert_model(**self.quote_tensors).pooler_output
            cls_n = self.bert_model(**batch_narrative_tensor).pooler_output

        elif self.bert_model_type == "distilbert-base-uncased":
            cls_q = self.bert_model(**self.quote_tensors).last_hidden_state
            cls_n = self.bert_model(**batch_narrative_tensor).last_hidden_state

            cls_q = cls_q[: , 0, :]
            cls_n = cls_n[: , 0, :]
        
            

        elif self.bert_model_type == "t5-base" or self.bert_model_type == "gpt2-base":
            cls_q = self.bert_model(**self.quote_tensors).last_hidden_state   # NQ x L X E
            cls_n = self.bert_model(**batch_narrative_tensor).last_hidden_state # B x L X E
            
            att_mask_quote = self.quote_tensors["attention_mask"].to(device)
            att_mask_narrative = batch_narrative_tensor["attention_mask"].to(device)
            att_mask_quote = att_mask_quote.view(att_mask_quote.size(0), att_mask_quote.size(1), 1 )
            att_mask_narrative = att_mask_narrative.view(att_mask_narrative.size(0), att_mask_narrative.size(1), 1 )
            att_mask_quote = att_mask_quote.repeat(1,1, cls_q.size(2))
            att_mask_narrative = att_mask_narrative.repeat(1,1, cls_n.size(2))

            
            
            cls_q = cls_q.mean(dim=1)
            cls_n = cls_n.mean(dim=1)
            
        elif self.bert_model_type == "bart-base":
            cls_q = self.bert_model(**self.quote_tensors).encoder_last_hidden_state   # NQ x L X E
            cls_n = self.bert_model(**batch_narrative_tensor).encoder_last_hidden_state # B x L X E
            att_mask_quote = self.quote_tensors["attention_mask"].to(device)
            att_mask_narrative = batch_narrative_tensor["attention_mask"].to(device)
            att_mask_quote = att_mask_quote.view(att_mask_quote.size(0), att_mask_quote.size(1), 1 )
            att_mask_narrative = att_mask_narrative.view(att_mask_narrative.size(0), att_mask_narrative.size(1), 1 )
            att_mask_quote = att_mask_quote.repeat(1,1, cls_q.size(2))
            att_mask_narrative = att_mask_narrative.repeat(1,1, cls_n.size(2))
            
            cls_q = cls_q * att_mask_quote
            cls_n = cls_n * att_mask_narrative
            
            cls_q = cls_q.mean(dim=1)
            cls_n = cls_n.mean(dim=1)

        elif self.bert_model_type == "sentencebert":
            def mean_pooling_sbert(model_output, attention_mask):
                token_embeddings = model_output[0] #First element of model_output contains all token embeddings
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            cls_q = self.bert_model(**self.quote_tensors)
            cls_n = self.bert_model(**batch_narrative_tensor)

            cls_q = mean_pooling_sbert(cls_q, self.quote_tensors['attention_mask'])
            cls_q = F.normalize(cls_q, p=2, dim=1) # NQ x E
            
            cls_n = mean_pooling_sbert(cls_n, batch_narrative_tensor['attention_mask'])
            cls_n = F.normalize(cls_n, p=2, dim=1) # B x E
        
        cls_q = torch.transpose(cls_q, 0, 1) # E X NQ
        
        logits = torch.matmul (cls_n, cls_q) # B X NQ
        
        return logits

    def evaluate(self, batch_narrative_tensor):
        if self.bert_model_type == "bert-base" or self.bert_model_type == "albert-base-v2":
            
            cls_q = self.bert_model(**self.quote_tensors).last_hidden_state
            cls_n = self.bert_model(**batch_narrative_tensor).last_hidden_state

            cls_q = cls_q[: , 0, :]
            cls_n = cls_n[: , 0, :]
        
        elif self.bert_model_type == "roberta-base":
            cls_q = self.bert_model(**self.quote_tensors).pooler_output
            cls_n = self.bert_model(**batch_narrative_tensor).pooler_output

        elif self.bert_model_type == "distilbert-base-uncased":
            # for distilbert, t5encoder, gpt2
            cls_q = self.bert_model(**self.quote_tensors).last_hidden_state
            cls_n = self.bert_model(**batch_narrative_tensor).last_hidden_state

            cls_q = cls_q[: , 0, :]
            cls_n = cls_n[: , 0, :]

            

        elif self.bert_model_type == "t5-base" or self.bert_model_type == "gpt2-base":
            cls_q = self.bert_model(**self.quote_tensors)
            cls_q = cls_q.last_hidden_state   # NQ x L X E
            cls_n = self.bert_model(**batch_narrative_tensor).last_hidden_state # B x L X E
            
            att_mask_quote = self.quote_tensors["attention_mask"].to(device)
            att_mask_narrative = batch_narrative_tensor["attention_mask"].to(device)
            att_mask_quote = att_mask_quote.view(att_mask_quote.size(0), att_mask_quote.size(1), 1 )
            att_mask_narrative = att_mask_narrative.view(att_mask_narrative.size(0), att_mask_narrative.size(1), 1 )
            att_mask_quote = att_mask_quote.repeat(1,1, cls_q.size(2))
            att_mask_narrative = att_mask_narrative.repeat(1,1, cls_n.size(2))

            
            
            cls_q = cls_q.mean(dim=1)
            cls_n = cls_n.mean(dim=1)

        elif self.bert_model_type == "bart-base":
            # for BART
            cls_q = self.bert_model(**self.quote_tensors).encoder_last_hidden_state   # NQ x L X E
            cls_n = self.bert_model(**batch_narrative_tensor).encoder_last_hidden_state # B x L X E
            att_mask_quote = self.quote_tensors["attention_mask"].to(device)
            att_mask_narrative = batch_narrative_tensor["attention_mask"].to(device)
            att_mask_quote = att_mask_quote.view(att_mask_quote.size(0), att_mask_quote.size(1), 1 )
            att_mask_narrative = att_mask_narrative.view(att_mask_narrative.size(0), att_mask_narrative.size(1), 1 )
            att_mask_quote = att_mask_quote.repeat(1,1, cls_q.size(2))
            att_mask_narrative = att_mask_narrative.repeat(1,1, cls_n.size(2))
            
            cls_q = cls_q * att_mask_quote
            cls_n = cls_n * att_mask_narrative

            

            cls_q = cls_q.mean(dim=1)
            cls_n = cls_n.mean(dim=1)
        
        elif self.bert_model_type == "sentencebert":
            def mean_pooling_sbert(model_output, attention_mask):
                token_embeddings = model_output[0] #First element of model_output contains all token embeddings
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            cls_q = self.bert_model(**self.quote_tensors)
            cls_n = self.bert_model(**batch_narrative_tensor)

            cls_q = mean_pooling_sbert(cls_q, self.quote_tensors['attention_mask'])
            cls_q = F.normalize(cls_q, p=2, dim=1) # NQ x E
            
            cls_n = mean_pooling_sbert(cls_n, batch_narrative_tensor['attention_mask'])
            cls_n = F.normalize(cls_n, p=2, dim=1) # B x E

        cls_q = torch.transpose(cls_q, 0, 1)
        logits = torch.matmul (cls_n, cls_q)
        
        logits = torch.softmax(logits, dim=-1)
        
        return logits

def get_prob_dist(model, model_type, all_inputs, debug=False):
    
    model.eval()
    pred_labels_return = []
    probs_return = []
    model.set_eval()
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    
    batch_size = 32
    num_batches = int(len(all_inputs)/batch_size)
    if len(all_inputs) % batch_size != 0:
        num_batches += 1

    matched = 0
    mrr = 0
    all_probs = []
    with torch.no_grad():
        for i in trange(num_batches):
            if i < num_batches - 1: 
                batch_input = all_inputs[ i* batch_size : (i+1) * batch_size]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_input = batch_input.to(device)
            else:
                batch_input = all_inputs[ i* batch_size :]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_input = batch_input.to(device)
                
            
            probs = model.evaluate(batch_input)
            probs = probs.detach().cpu().numpy()
            
            
            for p in probs:
                all_probs.append(p)
            
    all_probs = np.array(all_probs)
    
    return all_probs



def load_model(model, checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    return model

def tokenize_and_make_tensor(quotes, model_type, narratives=None):
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    
    
    qtok = bert_tokenizer(quotes, return_tensors="pt", padding=True, truncation=True, max_length = hyper_params["max_length_quote"])
    
    
    return qtok

def main(model_type, load_path, load_model_flag):
    TASK_DATA_PATH = "./data/data_splits/"

    train_pk = json.load(open(TASK_DATA_PATH + "task_1_proverb_only_seen_train_data_indices.json", "r"))
    
    test_pk = json.load(open(TASK_DATA_PATH + "task_1_proverb_only_seen_test_data_indices.json", "r"))
    label_map = json.load(open(TASK_DATA_PATH + "task_1_proverb_only_seen_label_map.json", "r"))
    label_map = {str(k):int(v) for k,v in label_map.items()}
    
    all_quotes = list(label_map.keys())

    assert train_pk != test_pk

    for i in range(len(all_quotes)):
        assert label_map[all_quotes[i]] == i

    dataset = json.load(open(hyper_params["dataset_file"], "r"))
    reverse_label_map = {int(v):str(k) for k,v in label_map.items()}
    

    train_data = []
    test_data = []

    for d in dataset:
        pk = d["pk"]
        quote = str(d["fields"]["quote"].strip("\n").strip().lower())
        narr = str(d["fields"]["narrative"].strip("\n").strip().lower())

        if pk in train_pk:
            assert pk not in test_pk
            train_data.append((narr.lower(), quote.lower(), label_map[quote.lower()] ))
        elif pk in test_pk:
            assert pk not in train_pk
            test_data.append((narr.lower(), quote.lower(), label_map[quote.lower()] ))
        else:
            ValueError

    shuffle(train_data)
    
    all_narratives_train = [x[0] for x in train_data]    
    labels_train = [int(x[2]) for x in train_data]
    all_narratives_test = [x[0] for x in test_data]    
    labels_test = [int(x[2]) for x in test_data]
 

    
    qtok = tokenize_and_make_tensor(all_quotes, model_type)
    labels_test_tensor = torch.from_numpy(np.array(labels_test)).long()
    
    qtok = qtok.to(device)
    labels_test_tensor = labels_test_tensor.to(device)
    
    model = Model(qtok, model_type)
    model = model.to(device)
    
    if load_model_flag:
        print("Loading fine-tuned model ... ")
        model = load_model(model, load_path)

    model.eval()

    prob_dist = get_prob_dist(model, model_type, all_narratives_test,debug=True) # N_test x 250
    assert prob_dist.shape[0] == len(all_narratives_test)
    assert prob_dist.shape[1] == 250

    similar_count = 0
    using_distance_metric = True

    if using_distance_metric:
        import scipy
        for distance_metric in ["cosine", "jsd", "euclidean", "cityblock"]:
            print("distance_metric = ", distance_metric)
            similar_count = 0
            for i in range(len(all_narratives_test)):
                min_dist = 100000
                min_index = -1
                for j in range(len(all_narratives_test)):
                    if i == j :
                        continue
                    ni = all_narratives_test[i]
                    nj = all_narratives_test[j]

                    pi = prob_dist[i]
                    pj = prob_dist[j]
                    if distance_metric == "jsd":
                        dist = jensenshannon(pi, pj)
                    elif distance_metric == "cosine":
                        dist = scipy.spatial.distance.cosine(pi, pj)
                    elif distance_metric == "euclidean":
                        dist = scipy.spatial.distance.euclidean(pi, pj)
                    elif distance_metric == "cityblock":
                        dist = scipy.spatial.distance.cityblock(pi, pj)
                    else:
                        raise NotImplementedError

                    if dist < min_dist:
                        min_dist = dist
                        min_index = j

                    
                li = labels_test[i]
                lj = labels_test[min_index]
                if li == lj:
                    
                    similar_count += 1


        
            print("Could find similar for : ", similar_count)
            print("Percent OR accuracy = ", float(similar_count) / float(len(all_narratives_test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='identify similar narrative')
    parser.add_argument('--model', type=str, choices = list(LLM_model.keys()), help='specify model name')    
    args = parser.parse_args()

    model_name = str(args.model).lower()
    if model_name not in list(LLM_model.keys()):
        raise ValueError("model not in list")

    log_filename = "./logs_identify_similar_narrative/{}_seen.log".format(model_name)

    logging.basicConfig(
        filename=log_filename, 
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Finetuning : {}".format(model_name))
    hyper_params["save_dir"] = None

    load_path = "./saved_models/{}_seen/best_model.pth".format(model_name)
    
    load_model_flag = True
    main(model_name, load_path, load_model_flag = load_model_flag)
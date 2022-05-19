import json
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AlbertModel, AlbertTokenizer
from transformers import BartTokenizer, BartModel
from transformers import T5Model, T5Tokenizer, T5EncoderModel
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AdamW, Adafactor
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
import operator 
import argparse
import logging

hyper_params = {
    "seed" : 42,
    "max_length_quote": 33,
    "max_length_narrative": 256,
    "batch_size": 8,
    "num_epochs": 5,
    "learning_rate": 3e-5,
    "train_test_split": 0.6,
    "dataset_file": "./data/full_dataset.json",
    "experiment_name" : "unseen_proverb_classification",
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
    "albert-base-v2" : AlbertModel.from_pretrained('albert-base-v2', cache_dir=CACHE_DIR),
    "bart-base" : BartModel.from_pretrained("facebook/bart-base", cache_dir=CACHE_DIR), 
    "t5-base" : T5EncoderModel.from_pretrained("t5-base", cache_dir=CACHE_DIR),
    "gpt2-base" : GPT2Model.from_pretrained("gpt2", cache_dir=CACHE_DIR),
    "sentencebert" : AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR)
}

LLM_tokenizer = {
    "bert-base" : BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR),
    "roberta-base" : RobertaTokenizer.from_pretrained("roberta-base", cache_dir=CACHE_DIR),
    "distilbert-base-uncased" : DistilBertTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_DIR),
    "albert-base-v2" : AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir=CACHE_DIR),
    "bart-base" : BartTokenizer.from_pretrained("facebook/bart-base", cache_dir=CACHE_DIR),
    "t5-base" : T5Tokenizer.from_pretrained("t5-base", cache_dir=CACHE_DIR),
    "gpt2-base" : GPT2Tokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR),
    "sentencebert" : AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR)
}


class ProverbModel(nn.Module):
    def __init__(self, quote_tensors, model_type):
        super(ProverbModel, self).__init__()
        self.bert_model_type = model_type
        self.bert_model = LLM_model[self.bert_model_type]
        
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
                token_embeddings = model_output[0] 
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
        probs, labels = torch.topk(logits,k=1, dim=-1) 
        _, ranks = torch.topk(logits,k=int(logits.size()[-1]), dim=-1)
        return probs, labels, ranks,logits


def save_model(model, opt, ep=None):
    path = hyper_params["save_dir"]
    model_save_checkpoint_filename = path + "best_model.pth"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({"model":model.state_dict(), "opt":opt.state_dict()}, model_save_checkpoint_filename)

def tester(model, model_type, all_inputs, labels, epoch_for_test=0, debug=False):
    model.eval()
    model.set_eval()

    pred_labels_return = []
    probs_return = []
    all_probs_return = []
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
    with torch.no_grad():
        for i in trange(num_batches):
            if i < num_batches - 1: 
                batch_input = all_inputs[ i* batch_size : (i+1) * batch_size]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_labels = labels[ i* batch_size : (i+1) * batch_size]
                batch_input = batch_input.to(device)
            else:
                batch_input = all_inputs[ i* batch_size :]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_input = batch_input.to(device)
                batch_labels = labels[ i* batch_size :]
                
            probs, preds, ranks, all_probs = model.evaluate(batch_input)
            # all_probs = all_probs.detach().cpu().numpy()
            all_probs_return.append(all_probs)
            # logging.info("pred = ", preds)
            matched_new = float(torch.sum(preds.view(-1) == batch_labels.view(-1)))
            tmp = batch_labels.view(-1, 1)
            targets_rank = tmp.expand_as(ranks)
            hits = (targets_rank == ranks).nonzero()
            rranks = torch.reciprocal((hits[:, -1] + 1).float())
            mrr += torch.sum(rranks)
            
            for pl in preds.view(-1):
                pred_labels_return.append(int(pl))
                
            for pl in probs.view(-1):
                probs_return.append(float(pl))
            # logging.info(matched_new)
            matched += matched_new
            # logging.info(matched)

            # logging.info("*****************\n\n")
            del batch_input

    logging.info("matched = {}".format(matched))
    logging.info("total = {}".format(float(len(all_inputs))))
    acc = float(matched) / float(len(all_inputs))
    mrr /= float(len(all_inputs))
    logging.info("Acc = {}".format(acc))
    
    logging.info("MRR = {}".format(mrr))
    
    if debug:
        all_probs_return = torch.cat(all_probs_return, dim = 0)
        all_probs_return = all_probs_return.detach().cpu().numpy()
        return pred_labels_return, probs_return, all_probs_return
    else:

        return acc, mrr

        

def trainer(model, model_type, optimizer, all_inputs, labels, epoch_for_train):
    model.train()
    model.set_train()
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    
    batch_size = hyper_params["batch_size"]

    num_batches = int(len(all_inputs)/batch_size)
    if len(all_inputs) % batch_size != 0:
        num_batches += 1

    criterion = torch.nn.CrossEntropyLoss()
    

    
    for ep in range(1):
        epoch_loss = 0
        for i in trange(num_batches):
            if i < num_batches - 1: 
                batch_input = all_inputs[ i* batch_size : (i+1) * batch_size]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_labels = labels[ i* batch_size : (i+1) * batch_size]
                batch_input = batch_input.to(device)
            else:
                batch_input = all_inputs[ i* batch_size :]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_input = batch_input.to(device)
                batch_labels = labels[ i* batch_size :]
                
            optimizer.zero_grad()
            model.zero_grad()

            logits = model(batch_input)
            
            loss = criterion(logits, batch_labels)
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            epoch_loss += loss.item() * len(batch_input)
            
            
            optimizer.step()
            
            del batch_input

        logging.info("Epoch {} avg loss = {}".format(epoch_for_train,float(epoch_loss)/float(len(all_inputs)) ) )
        
    
    return model, optimizer


def tokenize_and_make_tensor(quotes, model_type, narratives=None):
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    
    qtok = bert_tokenizer(quotes, return_tensors="pt", padding=True, truncation=True, max_length = hyper_params["max_length_quote"])
    

    
    return qtok

def main(model_type):
    TASK_DATA_PATH = "./data/data_splits/"

    train_pk = json.load(open(TASK_DATA_PATH + "task_1_proverb_only_unseen_train_data_indices.json", "r"))

    test_pk = json.load(open(TASK_DATA_PATH + "task_1_proverb_only_unseen_test_data_indices.json", "r"))

    test_quotes = []
    train_quotes = []

    dataset = json.load(open(hyper_params["dataset_file"], "r"))

    for d in dataset:
        pk = d["pk"]
        quote = str(d["fields"]["quote"].strip("\n").strip().lower())
        
        if pk in train_pk:
            assert pk not in test_pk
            if quote.lower() not in train_quotes:
                train_quotes.append(quote.lower())
        elif pk in test_pk:
            assert pk not in train_pk
            if quote.lower() not in test_quotes:
                test_quotes.append(quote.lower())
        else:
            ValueError

    for q in test_quotes:
        assert q not in train_quotes

    label_map = {}
    for q in train_quotes:
        if q.lower() not in list(label_map.keys()):
            label_map[q.lower()] = len(label_map)

    test_label_map = {}
    for q in test_quotes:
        if q.lower() not in list(test_label_map.keys()):
            test_label_map[q.lower()] = len(test_label_map)

    all_quotes = list(label_map.keys())

    assert train_pk != test_pk

    

    
    reverse_label_map = {int(v):str(k) for k,v in label_map.items()}
    reverse_test_label_map = {int(v):str(k) for k,v in test_label_map.items()}
    

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
            test_data.append((narr.lower(), quote.lower(), test_label_map[quote.lower()] ))
        else:
            ValueError

    shuffle(train_data)
    
    all_narratives_train = [x[0] for x in train_data]    
    labels_train = [int(x[2]) for x in train_data]
    all_narratives_test = [x[0] for x in test_data]    
    labels_test = [int(x[2]) for x in test_data]

    

    qtok = tokenize_and_make_tensor(train_quotes, model_type)
    qtok_test = tokenize_and_make_tensor(test_quotes, model_type)

    assert(len(qtok["input_ids"]) == 150 )
    assert(len(qtok_test["input_ids"]) == 100 )



    labels_train_tensor = torch.from_numpy(np.array(labels_train)).long()
    labels_test_tensor = torch.from_numpy(np.array(labels_test)).long()
    
    qtok = qtok.to(device)
    qtok_test = qtok_test.to(device)

    labels_train_tensor = labels_train_tensor.to(device)
    labels_test_tensor = labels_test_tensor.to(device)
    
   

    model = ProverbModel(qtok, model_type)
    model = model.to(device)
    

    # TODO: load a checkpoint below to test
    # checkpoint = torch.load("path_to_model")
    # model.load_state_dict(checkpoint["model"])

    model.train()
    model.set_train()

     
    param_optimizer = list(model.named_parameters())

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # optimizer = AdamW(optimizer_grouped_parameters, lr = hyper_params["learning_rate"], correct_bias=False)
    optimizer = AdamW(optimizer_grouped_parameters, lr = hyper_params["learning_rate"])
    
    # optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True, lr=None)  # for T5

    
    
    debug = False
    
    acc_list = []
    mrr_list = []

    best_acc = -1
    best_mrr = -1
    best_epoch = -1
    

    # TODO: Uncomment below line to test
    # pred_labels, probs, all_probs = tester(model, model_type, all_narratives_test, labels_test_tensor, epoch_for_test=0, debug=True)
    
    # the code below is for training
    start_epoch_id = 0
    for i in range(start_epoch_id, hyper_params["num_epochs"] + start_epoch_id):
        
        logging.info("Epoch main {}".format(i))
        model, optimizer = trainer(model, model_type, optimizer, all_narratives_train, labels_train_tensor, epoch_for_train=i)
        
        torch_random_state = torch.get_rng_state()
        model.quote_tensors = qtok_test
        _acc, _mrr = tester(model, model_type, all_narratives_test, labels_test_tensor, epoch_for_test=i, debug=False)
        model.quote_tensors = qtok
        torch.set_rng_state(torch_random_state)

        acc_list.append(_acc)
        mrr_list.append(_mrr)

        
        if _acc > best_acc:
            best_acc = _acc
            best_mrr = _mrr
            best_epoch = i
            save_model(model, optimizer)
        
        elif _acc == best_acc and _mrr > best_mrr:
            best_acc = _acc
            best_mrr = _mrr
            best_epoch = i
            save_model(model, optimizer)

    
    logging.info("Max acc =  {}".format(float(max(acc_list))))
    logging.info("Max mrr =  {}".format(float(max(acc_list))))

    logging.info("Max acc =  {}".format(best_acc))
    logging.info("Max mrr =  {}".format(best_mrr))
    logging.info("Best model saved at epoch = {}".format(best_epoch))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='proverb prediction')
    parser.add_argument('--model', type=str, choices = list(LLM_model.keys()), help='specify model name')    
    args = parser.parse_args()

    model_name = str(args.model).lower()
    if model_name not in list(LLM_model.keys()):
        raise ValueError("model not in list")

    log_filename = "./logs/{}_unseen.log".format(model_name)

    logging.basicConfig(
        filename=log_filename, 
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Finetuning : {}".format(model_name))
    hyper_params["save_dir"] = "./saved_models/{}_unseen/".format(model_name)
    main(model_name)
    
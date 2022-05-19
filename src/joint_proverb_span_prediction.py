import json
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AlbertModel, AlbertTokenizer
from transformers import BartTokenizer, BartModel
from transformers import T5Model, T5Tokenizer, T5EncoderModel
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AdamW
from transformers import Adafactor
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
    "max_length_narrative": 230,
    "batch_size": 2,
    "num_epochs": 30,
    "learning_rate": 5e-6,
    "train_test_split": 0.6,
    "dataset_file": "./data/full_dataset.json",
    "experiment_name" : "learned_roberta_2251_ep5_6040_mrr" 
}


random.seed(hyper_params["seed"])
np.random.seed(hyper_params["seed"])
torch.manual_seed(hyper_params["seed"])
torch.cuda.manual_seed(hyper_params["seed"])

device = torch.device("cuda")


CACHE_DIR = "./hf_cache/"

LLM_model = {
    "bert-base" : BertModel.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR),
    "roberta-base":RobertaModel.from_pretrained("roberta-base", cache_dir=CACHE_DIR)
    
}

LLM_tokenizer = {
    "bert-base" : BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR),
    "roberta-base" : RobertaTokenizer.from_pretrained("roberta-base", cache_dir=CACHE_DIR)
    
}

LLM_config = {
    "bert-base" : BertConfig.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR),
    "roberta-base" : RobertaConfig.from_pretrained("roberta-base", cache_dir=CACHE_DIR)
}

class Model(nn.Module):
    def __init__(self, quote_tensors, model_type):
        super(Model, self).__init__()
        self.bert_model_type = model_type
        self.bert_model = LLM_model[self.bert_model_type]
        config = LLM_config[self.bert_model_type]
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.bert_model = self.bert_model.to(device)
        self.quote_tensors = quote_tensors

        
        self.narr_linear = nn.Linear(2*768 , 768)
        self.start_linear = nn.Linear(768 , 1)
        self.end_linear = nn.Linear(768 , 1)
        
        
        self.tanh = nn.Tanh()
        self.narr_linear.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.narr_linear.bias is not None:
            self.narr_linear.bias.data.zero_()

        self.start_linear.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.start_linear.bias is not None:
            self.start_linear.bias.data.zero_()
        
        self.end_linear.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.end_linear.bias is not None:
            self.end_linear.bias.data.zero_()

    def set_train(self):
        self.bert_model.train()
        self.train()
        
       
        

    def set_eval(self):
        self.bert_model.eval()
        self.eval()

    def forward(self,batch_narrative_tensor, batch_span_tensor):
        
        
        if self.bert_model_type == "bert-base" or self.bert_model_type == "albert-base-v2":
            cls_q = self.bert_model(**self.quote_tensors).last_hidden_state
            cls_q = cls_q[: , 0, :]

            cls_q_span = self.bert_model(**batch_span_tensor).last_hidden_state
            cls_q_span = cls_q_span[: , 0, :]
            
            out2 = self.bert_model(**batch_narrative_tensor)

            cls_n = out2.last_hidden_state
            cls_n = cls_n[: , 0, :]
            cls_n_all = out2.last_hidden_state
            batch_size = cls_n_all.size(0)
            seq_len = cls_n_all.size(1)

            cls_q_span_repeated = cls_q_span.unsqueeze(1).repeat(1,cls_n_all.size(1), 1)
            
            cls_n_all = torch.cat([cls_q_span_repeated, cls_n_all], dim=-1)
            cls_n_all_compressed = self.narr_linear(cls_n_all)

            start_logits = self.start_linear(cls_n_all_compressed).view(batch_size, seq_len)
            end_logits = self.end_linear(cls_n_all_compressed).view(batch_size, seq_len)
            
        
        elif self.bert_model_type == "roberta-base":
            cls_q = self.bert_model(**self.quote_tensors).pooler_output
            
            cls_q_span = self.bert_model(**batch_span_tensor).pooler_output
            
            out2 = self.bert_model(**batch_narrative_tensor)

            cls_n = out2.pooler_output
            cls_n_all = out2.last_hidden_state
            batch_size = cls_n_all.size(0)
            seq_len = cls_n_all.size(1)

            cls_q_span_repeated = cls_q_span.unsqueeze(1).repeat(1,cls_n_all.size(1), 1)
            
            cls_n_all = torch.cat([cls_q_span_repeated, cls_n_all], dim=-1)
            cls_n_all_compressed = self.narr_linear(cls_n_all)

            start_logits = self.start_linear(cls_n_all_compressed).view(batch_size, seq_len)
            end_logits = self.end_linear(cls_n_all_compressed).view(batch_size, seq_len)


        
        cls_q = torch.transpose(cls_q, 0, 1) # E X NQ
        logits = torch.matmul (cls_n, cls_q) # B X NQ
        
        return logits, start_logits, end_logits

    def evaluate(self, batch_narrative_tensor, batch_span_tensor):
        
        if self.bert_model_type == "bert-base" or self.bert_model_type == "albert-base-v2":
            cls_q = self.bert_model(**self.quote_tensors).last_hidden_state
            cls_q = cls_q[: , 0, :]

            cls_q_span = self.bert_model(**batch_span_tensor).last_hidden_state
            cls_q_span = cls_q_span[: , 0, :]
            
            out2 = self.bert_model(**batch_narrative_tensor)

            cls_n = out2.last_hidden_state
            cls_n = cls_n[: , 0, :]
            cls_n_all = out2.last_hidden_state
            batch_size = cls_n_all.size(0)
            seq_len = cls_n_all.size(1)

            cls_q_span_repeated = cls_q_span.unsqueeze(1).repeat(1,cls_n_all.size(1), 1)
            
            cls_n_all = torch.cat([cls_q_span_repeated, cls_n_all], dim=-1)
            cls_n_all_compressed = self.narr_linear(cls_n_all)

            start_logits = self.start_linear(cls_n_all_compressed).view(batch_size, seq_len)
            end_logits = self.end_linear(cls_n_all_compressed).view(batch_size, seq_len)
            
        
        elif self.bert_model_type == "roberta-base" :
            cls_q = self.bert_model(**self.quote_tensors).pooler_output
            
            cls_q_span = self.bert_model(**batch_span_tensor).pooler_output
            
            out2 = self.bert_model(**batch_narrative_tensor)

            cls_n = out2.pooler_output
            cls_n_all = out2.last_hidden_state
            batch_size = cls_n_all.size(0)
            seq_len = cls_n_all.size(1)

            cls_q_span_repeated = cls_q_span.unsqueeze(1).repeat(1,cls_n_all.size(1), 1)
            
            cls_n_all = torch.cat([cls_q_span_repeated, cls_n_all], dim=-1)
            cls_n_all_compressed = self.narr_linear(cls_n_all)

            start_logits = self.start_linear(cls_n_all_compressed).view(batch_size, seq_len)
            end_logits = self.end_linear(cls_n_all_compressed).view(batch_size, seq_len)
        

        cls_q = torch.transpose(cls_q, 0, 1)
        logits = torch.matmul (cls_n, cls_q)
        logits = torch.softmax(logits, dim=-1)
        start_logits = torch.softmax(start_logits, dim=-1)
        end_logits = torch.softmax(end_logits, dim=-1)

        probs, labels = torch.topk(logits,k=1, dim=-1) 
        _, ranks = torch.topk(logits,k=int(logits.size()[-1]), dim=-1)
        return probs, labels, ranks,logits, start_logits, end_logits


def save_model(model, opt):
    path = hyper_params["save_dir"]
    model_save_checkpoint_filename = path + "best_model.pth"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({"model":model.state_dict(), "opt":opt.state_dict()}, model_save_checkpoint_filename)

def tester(model, model_type, all_inputs,  all_span_inputs, all_span_labels, labels, epoch_for_test=0, debug=False):
    
    model.eval()
    pred_labels_return = []
    probs_return = []
    all_probs_return = []
    model.set_eval()
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base" or model_type == "distilgpt2":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    
    # form batches
    batch_size = 32
    num_batches = int(len(all_inputs)/batch_size)
    if len(all_inputs) % batch_size != 0:
        num_batches += 1

    matched = 0
    mrr = 0

    narratives_matched = set()
    span_score_map = {}

    start_labels = []
    end_labels = []
    cnt_error = 0
    for i in range(len(all_span_inputs)):
        # tokenize narrative
        _narr = bert_tokenizer.tokenize(" " + all_inputs[i])

        # tokenize span
        _span = bert_tokenizer.tokenize(" " + all_span_labels[i])
        
        try: 
            s,e = find_sub_list(_span,_narr )
        except:
            try: 
                s,e = find_sub_list(_span[1:-1],_narr )
                s -= 1
                e += 1
            except:
                cnt_error += 1
                s = 0
                e = len(_narr)
                

        s += 1 # take cls into account
        e += 1

        if e >= hyper_params["max_length_narrative"]:
            e = hyper_params["max_length_narrative"] - 1
        
        if e < s:
            e = s

        start_labels.append(s)
        end_labels.append(e)

    print("cnt_error = ", cnt_error)
    start_labels = torch.tensor(start_labels).long().to(device)
    end_labels = torch.tensor(end_labels).long().to(device)
    global_match = 0
    with torch.no_grad():
        for i in trange(num_batches):
            if i < num_batches - 1: 
                batch_input = all_inputs[ i* batch_size : (i+1) * batch_size]
                batch_input_copy = batch_input.copy()
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_labels = labels[ i* batch_size : (i+1) * batch_size]
                batch_input = batch_input.to(device)
                batch_span_input = all_span_inputs[ i* batch_size : (i+1) * batch_size]
                batch_span_input = bert_tokenizer(batch_span_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_quote"])
                batch_span_input = batch_span_input.to(device)
                batch_start_labels = start_labels[i* batch_size : (i+1) * batch_size]
                batch_end_labels = end_labels[i* batch_size : (i+1) * batch_size]
               
            else:
                batch_input = all_inputs[ i* batch_size :]
                batch_input_copy = batch_input.copy()
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_span_input = all_span_inputs[ i* batch_size : ]
                batch_span_input = bert_tokenizer(batch_span_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_quote"])
               
                batch_input = batch_input.to(device)
                batch_labels = labels[ i* batch_size :]
                batch_span_input = batch_span_input.to(device)
                batch_start_labels = start_labels[i* batch_size : ]
                batch_end_labels = end_labels[i* batch_size : ]

                
            probs, preds, ranks, all_probs, start_logits, end_logits = model.evaluate(batch_input, batch_span_input)
            
            start_logits *= batch_input["attention_mask"]
            end_logits *= batch_input["attention_mask"]


            all_probs_return.append(all_probs)
            
            preds = preds.view(-1)
            batch_labels = batch_labels.view(-1)
            for jkj in range(len(preds)):
                proverb_correctly_predicted = False
                if float(preds[jkj]) == float(batch_labels[jkj]):
                    proverb_correctly_predicted = True

                if batch_input_copy[jkj] not in narratives_matched:
                    if float(preds[jkj]) == float(batch_labels[jkj]):
                        global_match += 1
                        proverb_correctly_predicted = True
                    narratives_matched.add(batch_input_copy[jkj])
                


                start_pos = start_logits[jkj].view(-1).argmax()
                end_pos = end_logits[jkj].view(-1).argmax()

                if end_pos < start_pos:
                    pred_span = "<nan>"
                else:
                    if end_pos == start_pos:
                        end_pos = start_pos + 1
                    pred_span = bert_tokenizer.decode(batch_input["input_ids"][jkj][start_pos:end_pos] , skip_special_tokens=True, clean_up_tokenization_spaces=True)

                true_span = all_span_labels[jkj]
                pred_span = pred_span.strip()
                if len(pred_span) == 0:
                    pred_span = "<nan>"
                pred_span_token_set = set(pred_span.split())
                true_span_token_set = set(true_span.split())
                inter_set = true_span_token_set.intersection(pred_span_token_set)
                
                # print("pred span = ", pred_span_token_set)
                prec_single = float(len(inter_set)) / float(len(pred_span_token_set))
                recall_single = float(len(inter_set)) / float(len(true_span_token_set))
                
                if batch_input_copy[jkj] not in span_score_map.keys():
                    span_score_map[batch_input_copy[jkj]] = {
                        "r" : [],
                        "p" : [],
                        "prov" : []
                        # "f" : []
                    }

                
                span_score_map[batch_input_copy[jkj]]["r"].append(recall_single)
                span_score_map[batch_input_copy[jkj]]["p"].append(prec_single)
                if proverb_correctly_predicted:
                    span_score_map[batch_input_copy[jkj]]["prov"].append(1)
                else:
                    span_score_map[batch_input_copy[jkj]]["prov"].append(0)

                

            tmp = batch_labels.view(-1, 1)
            targets_rank = tmp.expand_as(ranks)
            hits = (targets_rank == ranks).nonzero()
            rranks = torch.reciprocal((hits[:, -1] + 1).float())
            mrr += torch.sum(rranks)
            
            for pl in preds.view(-1):
                pred_labels_return.append(int(pl))
                
            for pl in probs.view(-1):
                probs_return.append(float(pl))
            
            del batch_input

    
    global_recall = 0
    global_prec = 0
    
    for kkey in span_score_map.keys():
        global_recall += np.mean( np.multiply(np.array(span_score_map[kkey]["r"]) , np.array(span_score_map[kkey]["prov"]) )  )
        global_prec += np.mean( np.multiply(np.array(span_score_map[kkey]["p"]) , np.array(span_score_map[kkey]["prov"]) )  )
        

    global_recall /= len(span_score_map.keys())
    global_prec /= len(span_score_map.keys())
    

    global_f1 = 2 * global_recall * global_prec / (global_recall + global_prec)

    global_acc = float(global_match)/ float(len(narratives_matched))
    print("global_prec = ", global_prec)
    print("global_recall = ", global_recall)
    print("global_f1 = ", global_f1)
    print("global accuracy = ",global_acc)

    debug = False
    if debug:
        all_probs_return = torch.cat(all_probs_return, dim = 0)
        all_probs_return = all_probs_return.detach().cpu().numpy()
        return pred_labels_return, probs_return, all_probs_return
    else:

        return global_acc, global_recall, global_prec, global_f1

        
def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

def trainer(model, model_type, optimizer, all_inputs, all_span_inputs, all_span_labels, labels, epoch_for_train):

    model.train()
    model.set_train()
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base" or model_type == "distilgpt2":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    
    batch_size = hyper_params["batch_size"]

    num_batches = int(len(all_inputs)/batch_size)
    if len(all_inputs) % batch_size != 0:
        num_batches += 1

    criterion = torch.nn.CrossEntropyLoss()
    criterionSpan = torch.nn.CrossEntropyLoss(ignore_index=-100)
    

    start_labels = []
    end_labels = []
    cnt_error = 0
    for i in range(len(all_span_inputs)):
        _narr = bert_tokenizer.tokenize(" " + all_inputs[i])

        _span = bert_tokenizer.tokenize(" " + all_span_labels[i])
        
        try: 
            s,e = find_sub_list(_span,_narr )
        except:
            try: 
                s,e = find_sub_list(_span[1:-1],_narr )
                s -= 1
                e += 1
            except:
                cnt_error += 1
                s = 0
                e = len(_narr)
                

        s += 1 # take cls into account
        e += 1

        if e >= hyper_params["max_length_narrative"]:
            e = hyper_params["max_length_narrative"] - 1
        
        if e < s:
            e = s

        start_labels.append(s)
        end_labels.append(e)

    
    start_labels = torch.tensor(start_labels).long().to(device)
    end_labels = torch.tensor(end_labels).long().to(device)

    # loop over epochs
    for ep in range(1):
        # get batch
        epoch_loss = 0
        for i in trange(num_batches):
            if i < num_batches - 1: 
                batch_input = all_inputs[ i* batch_size : (i+1) * batch_size]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                
                batch_span_input = all_span_inputs[ i* batch_size : (i+1) * batch_size]
                batch_span_input = bert_tokenizer(batch_span_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_quote"])
                
                batch_labels = labels[ i* batch_size : (i+1) * batch_size]
                batch_start_labels = start_labels[i* batch_size : (i+1) * batch_size]
                batch_end_labels = end_labels[i* batch_size : (i+1) * batch_size]
                batch_input = batch_input.to(device)
                batch_span_input = batch_span_input.to(device)

            else:
                batch_input = all_inputs[ i* batch_size :]
                batch_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_narrative"])
                batch_span_input = all_span_inputs[ i* batch_size : ]
                batch_span_input = bert_tokenizer(batch_span_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_quote"])
                
                batch_input = batch_input.to(device)
                batch_span_input = batch_span_input.to(device)
                batch_start_labels = start_labels[i* batch_size : ]
                batch_end_labels = end_labels[i* batch_size : ]
                batch_labels = labels[ i* batch_size :]
                
            
            optimizer.zero_grad()
            logits, start_logits, end_logits = model(batch_input, batch_span_input)

            
            attn_map = batch_input["attention_mask"]
            inverse_attn_map = torch.ones(attn_map.size()).to(device) - attn_map
            inverse_attn_map = -100 * inverse_attn_map
            start_logits = (start_logits * attn_map) + inverse_attn_map
            end_logits = (end_logits * attn_map) + inverse_attn_map



            loss_proverb = criterion(logits, batch_labels)
            loss_start = criterionSpan(start_logits, batch_start_labels)
            loss_end = criterionSpan(end_logits, batch_end_labels)

            

            loss = loss_proverb + loss_start + loss_end

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_loss += loss.item() * len(batch_input)
            
            optimizer.step()
            del batch_input

        print("Epoch ", epoch_for_train, " avg loss = ", float(epoch_loss)/float(len(all_inputs)))
        
    return model, optimizer


def tokenize_and_make_tensor(quotes, model_type, narratives=None):
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base" or model_type == "distilgpt2":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    
    qtok = bert_tokenizer(quotes, return_tensors="pt", padding=True, truncation=True, max_length = hyper_params["max_length_quote"])
    

    
    return qtok

def main(model_type):
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

        sq1 = str(d["fields"]["span_quote_1"].strip("\n").strip().lower())
        sq2 = str(d["fields"]["span_quote_2"].strip("\n").strip().lower())
        sq3 = str(d["fields"]["span_quote_3"].strip("\n").strip().lower())
        sq4 = str(d["fields"]["span_quote_4"].strip("\n").strip().lower())
        sq5 = str(d["fields"]["span_quote_5"].strip("\n").strip().lower())

        sn1 = str(d["fields"]["span_narrative_1"].strip("\n").strip().lower())
        sn2 = str(d["fields"]["span_narrative_2"].strip("\n").strip().lower())
        sn3 = str(d["fields"]["span_narrative_3"].strip("\n").strip().lower())
        sn4 = str(d["fields"]["span_narrative_4"].strip("\n").strip().lower())
        sn5 = str(d["fields"]["span_narrative_5"].strip("\n").strip().lower())
        span_count = 0
        span_map_single_example = {}
        if len(sq1) > 0:
            span_count +=1
            span_map_single_example[sq1] = sn1
        if len(sq2) > 0:
            span_count +=1
            span_map_single_example[sq2] = sn2
        if len(sq3) > 0:
            span_count +=1
            span_map_single_example[sq3] = sn3
        if len(sq4) > 0:
            span_count +=1
            span_map_single_example[sq4] = sn4
        if len(sq5) > 0:
            span_count +=1
            span_map_single_example[sq5] = sn5

        if pk in train_pk:
            assert pk not in test_pk
            # train_data.append((narr.lower(), quote.lower(), label_map[quote.lower()] ))
            for sq_key in span_map_single_example.keys():
                train_data.append((narr.lower().lower(), quote.lower(), label_map[quote.lower()], sq_key, span_map_single_example[sq_key] ))


        elif pk in test_pk:
            assert pk not in train_pk
            # test_data.append((narr.lower(), quote.lower(), label_map[quote.lower()] ))
            for sq_key in span_map_single_example.keys():
                test_data.append((narr.lower().lower(), quote.lower(), label_map[quote.lower()], sq_key, span_map_single_example[sq_key] ))

        else:
            ValueError


    shuffle(train_data)
    
    all_narratives_train = [x[0] for x in train_data]
    all_spans_train = [x[3] for x in train_data]
    all_spans_train_label =  [x[4] for x in train_data]
    labels_train = [x[2] for x in train_data]


    all_narratives_test = [x[0] for x in test_data]    
    all_spans_test =  [x[3] for x in test_data]
    all_spans_test_label =  [x[4] for x in test_data]
    labels_test = [x[2] for x in test_data]


    
    qtok = tokenize_and_make_tensor(all_quotes, model_type)

    

    labels_train_tensor = torch.from_numpy(np.array(labels_train)).long()
    labels_test_tensor = torch.from_numpy(np.array(labels_test)).long()
    
    qtok = qtok.to(device)

    
    labels_train_tensor = labels_train_tensor.to(device)
    labels_test_tensor = labels_test_tensor.to(device)
    
    model = Model(qtok, model_type)
    model = model.to(device)
    
    # NOTE:give appropriate paths
    print("Loading proverb_model from " + "./saved_models/{}_seen/best_model.pth".format(model_type))
    checkpoint_proverb = torch.load("./saved_models/{}_seen/best_model.pth".format(model_type))
    model.load_state_dict(checkpoint_proverb["model"], strict=False)

    
    model.train()
    model.set_train()

    

    param_optimizer = list(model.named_parameters())

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    param_optimizer = [n for n in param_optimizer if n[1].requires_grad == True]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = hyper_params["learning_rate"], correct_bias=False)
    
    

    debug = False
    
    acc_list = []
    f1_list = []

    best_acc = -1
    best_f1 = -1
    best_epoch = -1
    
    # NOTE: uncoment below line if testing only and comment code below that
    # pred_labels, probs, all_probs = tester(model, model_type, all_narratives_test, all_spans_test, all_spans_test_label, labels_test_tensor, epoch_for_test=0, debug=True)
    
    
    
    # NOTE: code below is for training
    start_epoch_id = 0

    for i in range(start_epoch_id, hyper_params["num_epochs"] + start_epoch_id):
        
        print("Epoch main ", i)
        model, optimizer = trainer(model, model_type, optimizer, all_narratives_train, all_spans_train, all_spans_train_label, labels_train_tensor, epoch_for_train=i)

        torch_random_state = torch.get_rng_state()
        _acc, _recall, _prec, _f1 = tester(model, model_type, all_narratives_test, all_spans_test, all_spans_test_label, labels_test_tensor, epoch_for_test=0, debug=True)
        torch.set_rng_state(torch_random_state)
        

        if _acc > best_acc:
            best_acc = _acc
            best_f1 = _f1
            best_epoch = i
            save_model(model, optimizer)
        
        elif _acc == best_acc and _f1 > best_f1:
            best_acc = _acc
            best_mrr = _f1
            best_f1 = i
            save_model(model, optimizer)


    logging.info("Max acc =  {}".format(best_acc))
    logging.info("Max f1 =  {}".format(best_f1))
    logging.info("Best model saved at epoch = {}".format(best_epoch))
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='joint proverb and span prediction')
    parser.add_argument('--model', type=str, choices = list(LLM_model.keys()), help='specify model name')    
    args = parser.parse_args()

    model_name = str(args.model).lower()
    if model_name not in list(LLM_model.keys()):
        raise ValueError("model not in list")

    log_filename = "./logs_joint_model/{}_seen.log".format(model_name)

    logging.basicConfig(
        filename=log_filename, 
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Finetuning : {}".format(model_name))
    hyper_params["save_dir"] = "./saved_models_joint_learning/{}_seen/".format(model_name)
    main(model_name)
    
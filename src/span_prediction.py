import json
from transformers import BertModel, BertConfig, BertTokenizer, BertConfig
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
import logging
import argparse

hyper_params = {
    "seed" : 42,
    "max_length_quote": 33,
    "max_length_narrative": 230,
    "max_length_merged_context": 275,
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-5,
    "train_test_split": 0.6,
    "dataset_file": "./data/full_dataset.json",
    "experiment_name" : "seen_span_prediction" 
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

class SpanModel(nn.Module):
    def __init__(self, quote_tensors, model_type):
        super(SpanModel, self).__init__()
        self.bert_model_type = model_type
        self.bert_model = LLM_model[self.bert_model_type]
        self.bert_model = self.bert_model.to(device)
        config = LLM_config[self.bert_model_type]
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.start_linear = nn.Linear(config.hidden_size,2)
        self.end_linear = nn.Linear(config.hidden_size, 2)
        self.span_outputs = nn.Linear(config.hidden_size, 2)

        self.span_outputs.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.span_outputs.bias is not None:
            self.span_outputs.bias.data.zero_()

    def set_train(self):
        self.bert_model.train()
        self.train()

    def set_eval(self):
        self.bert_model.eval()
        self.eval()

    def forward(self,batch_merged_tensor):
        
        
        if self.bert_model_type == "bert-base" or self.bert_model_type == "roberta-base" or self.bert_model_type == "albert-base-v2":
            sequence_output = self.bert_model(**batch_merged_tensor).last_hidden_state      
            logits = self.span_outputs(sequence_output)

            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            
        return start_logits, end_logits

    def evaluate(self, batch_merged_tensor):
        
        if self.bert_model_type == "bert-base" or self.bert_model_type == "roberta-base" or self.bert_model_type == "albert-base-v2":
            sequence_output = self.bert_model(**batch_merged_tensor).last_hidden_state      
            logits = self.span_outputs(sequence_output)

            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

        start_logits = torch.softmax(start_logits, dim=-1)
        end_logits = torch.softmax(end_logits, dim=-1)

        return start_logits, end_logits


def post_process_predictions(all_start_logits, all_end_logits, narr_mask):
    # return batch-wise recall and precision and F1
    
    min_null_prediction = None
    predictions = []
    batch_size = all_start_logits.size(0)
    
    n_best_size = 5

    all_start_logits = all_start_logits.detach().cpu().numpy()
    all_end_logits = all_end_logits.detach().cpu().numpy()

    for b_idx in range(batch_size):
        prelim_predictions = []
        start_logits = all_start_logits[b_idx]
        end_logits = all_end_logits[b_idx]
        
        start_logits[0] = -1000
        end_logits[0] = -1000
        narr_length = int(narr_mask[b_idx].sum())

        start_logits[narr_length:] = -1000
        end_logits[narr_length:] = -1000

        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

        
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                prelim_predictions.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                        "start_index": start_index,
                        "end_index": end_index
                    }
                    )

        predictions.append(sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[0])
    
    assert len(predictions) == batch_size
    return predictions

def save_model(model, opt, monitoring):
    path = hyper_params["save_dir"]
    model_save_checkpoint_filename = path + "best_model_monitoring_{}.pth".format(monitoring)
    if not os.path.exists(path):
        os.makedirs(path)
    # torch.save({"model":model.state_dict(), "opt":opt.state_dict()}, path + "model_opt_ep_{}.pth".format(ep))
    torch.save({"model":model.state_dict(), "opt":opt.state_dict()}, model_save_checkpoint_filename)

def tester(model, model_type, all_pk, all_inputs, all_quotes_inputs, all_span_inputs, all_span_labels, epoch_for_test=0, debug=False):
    assert len(set(all_pk)) == 1000
    
    model.eval()
    model.set_eval()
    
    pred_labels_return = []
    probs_return = []
    all_probs_return = []
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
        # tokenize quote
        _quote = bert_tokenizer.tokenize(" " + all_quotes_inputs[i])
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

    logging.info("cnt_error = {}".format(cnt_error))
    start_labels = torch.tensor(start_labels).long().to(device)
    end_labels = torch.tensor(end_labels).long().to(device)
    
    global_recall_map = {} # pk : [] for each span list there recall
    global_precision_map = {}
    global_f1_map = {}

    with torch.no_grad():
        for i in trange(num_batches):
            if i < num_batches - 1: 
                batch_pk = all_pk[i* batch_size : (i+1) * batch_size]
                batch_input = all_inputs[ i* batch_size : (i+1) * batch_size]
                batch_quotes_input = all_quotes_inputs[ i* batch_size : (i+1) * batch_size]
                batch_span_input = all_span_inputs[ i* batch_size : (i+1) * batch_size]
                
                batch_merged_input = [ "[CLS] " +  batch_input[k_id] + " [SEP] " + batch_quotes_input[k_id] + " [SEP] " + batch_span_input[k_id] for k_id in range(len(batch_input)) ]
                
                batch_merged_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_merged_context"])
                
                batch_input_copy = batch_input.copy()
                
                batch_start_labels = start_labels[i* batch_size : (i+1) * batch_size]
                batch_end_labels = end_labels[i* batch_size : (i+1) * batch_size]
                batch_merged_input = batch_merged_input.to(device)

                

            else:
                batch_pk = all_pk[i* batch_size :]
                batch_input = all_inputs[ i* batch_size :]
                batch_quotes_input = all_quotes_inputs[ i* batch_size : ]
                batch_span_input = all_span_inputs[ i* batch_size : ]
                
                batch_merged_input = [ "[CLS] " +  batch_input[k_id] + " [SEP] " + batch_quotes_input[k_id] + " [SEP] " + batch_span_input[k_id] for k_id in range(len(batch_input)) ]
                
                batch_merged_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_merged_context"])
                batch_merged_input = batch_merged_input.to(device)
                
                batch_input_copy = batch_input.copy()
                
                batch_start_labels = start_labels[i* batch_size : ]
                batch_end_labels = end_labels[i* batch_size : ]

            # prepare narrative mask
            narr_mask = torch.zeros(batch_merged_input["input_ids"].size())
            for k_id in range(int(batch_merged_input["input_ids"].size(0))):
                _narr_length = len(batch_input[k_id])
                narr_mask[k_id][:_narr_length+1] = 1  # include CLS

            narr_mask = narr_mask.to(device)

            # get loss
            # print(batch_input)
            start_logits, end_logits = model.evaluate(batch_merged_input)
            
            start_logits *= narr_mask
            end_logits *= narr_mask

            predictions = post_process_predictions(start_logits, end_logits, narr_mask)

            # calculate prec, recall
            for b_idx in range(narr_mask.size(0)):
                try:
                    prediction = predictions[b_idx]
                except:
                    logging.info("ERROR")
                    logging.info("len predictions = {}".format(len(predictions)))
                    logging.info("narr_mask = {}".format(narr_mask.size()))
                    sys.exit()

                predicted_span = batch_merged_input["input_ids"][b_idx][prediction["start_index"] : 1 + prediction["end_index"] ]
                try: 
                    predicted_span = bert_tokenizer.decode(predicted_span)
                except:
                    logging.info("ERROR")
                    logging.info("prediction = {}".format(prediction))
                    predicted_span = batch_merged_input["input_ids"][b_idx][prediction["start_index"] : 1 + prediction["end_index"] ]
                    logging.info("predicted_span = {}".format(predicted_span))
                    sys.exit()

                pred_tokens = set(predicted_span.split())
                gold_tokens = set(all_span_labels[b_idx].split())
                intersection_set = gold_tokens.intersection(pred_tokens)
                
                try: 
                    precision = float(len(intersection_set)) / float(len(pred_tokens))
                except:
                    precision = 0

                try:
                    recall = float(len(intersection_set)) / float(len(gold_tokens))
                    # print("Predicted span = ", predicted_span)
                    # print("Gold span = ", all_span_labels[b_idx])
                    # print("Recall = ", recall)
                except:
                    recall = 0

                if batch_pk[b_idx] not in global_recall_map.keys():
                    global_recall_map[batch_pk[b_idx]] = []

                global_recall_map[batch_pk[b_idx]].append(recall)

                if batch_pk[b_idx] not in global_precision_map.keys():
                    global_precision_map[batch_pk[b_idx]] = []

                global_precision_map[batch_pk[b_idx]].append(precision)
                
    avg_recall = np.mean(np.array([ np.mean(np.array(v)) for k,v in global_recall_map.items() ]))
    assert len(global_recall_map.keys()) == 1000
    assert len(global_precision_map.keys()) == 1000
    avg_precision = np.mean(np.array([ np.mean(np.array(v)) for k,v in global_precision_map.items() ]))
    logging.info("avg_recall = {}".format(avg_recall))
    logging.info("avg_precision = {}".format(avg_precision))
    
    return avg_recall, avg_precision

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

def trainer(model, model_type, optimizer, all_inputs, all_quotes_inputs, all_span_inputs, all_span_labels, epoch_for_train):
    model.train()
    model.set_train()
    bert_tokenizer = LLM_tokenizer[model_type]
    if model_type == "gpt2-base" or model_type == "distilgpt2":
        bert_tokenizer.pad_token = bert_tokenizer.eos_token
    # form batches
    batch_size = hyper_params["batch_size"]

    num_batches = int(len(all_inputs)/batch_size)
    if len(all_inputs) % batch_size != 0:
        num_batches += 1

    # criterion = torch.nn.CrossEntropyLoss()
    criterionSpan = torch.nn.CrossEntropyLoss(ignore_index=-100)
    

    # convert tokens to start and end id
    start_labels = []
    end_labels = []
    cnt_error = 0
    for i in range(len(all_span_inputs)):
        _narr = bert_tokenizer.tokenize(" " + all_inputs[i])
        _quote = bert_tokenizer.tokenize(" " + all_quotes_inputs[i])
        
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

    logging.info("cnt_error = {}".format(cnt_error))
    start_labels = torch.tensor(start_labels).long().to(device)
    end_labels = torch.tensor(end_labels).long().to(device)

    # loop over epochs
    for ep in range(1):
        # get batch
        epoch_loss = 0
        for i in trange(num_batches):
            if i < num_batches - 1: 
                batch_input = all_inputs[ i* batch_size : (i+1) * batch_size]
                batch_quotes_input = all_quotes_inputs[ i* batch_size : (i+1) * batch_size]
                batch_span_input = all_span_inputs[ i* batch_size : (i+1) * batch_size]
                
                batch_merged_input = [ "[CLS] " +  batch_input[k_id] + " [SEP] " + batch_quotes_input[k_id] + " [SEP] " + batch_span_input[k_id] for k_id in range(len(batch_input)) ]
                
                batch_merged_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_merged_context"])
                
                batch_start_labels = start_labels[i* batch_size : (i+1) * batch_size].to(device)
                batch_end_labels = end_labels[i* batch_size : (i+1) * batch_size].to(device)
                batch_merged_input = batch_merged_input.to(device)
                

            else:
                batch_input = all_inputs[ i* batch_size :]
                batch_quotes_input = all_quotes_inputs[ i* batch_size : ]
                batch_span_input = all_span_inputs[ i* batch_size : ]
                
                batch_merged_input = [ "[CLS] " +  batch_input[k_id] + " [SEP] " + batch_quotes_input[k_id] + " [SEP] " + batch_span_input[k_id] for k_id in range(len(batch_input)) ]
                
                batch_merged_input = bert_tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=hyper_params["max_length_merged_context"])
                
                batch_merged_input = batch_merged_input.to(device)
                batch_start_labels = start_labels[i* batch_size : ].to(device)
                batch_end_labels = end_labels[i* batch_size : ].to(device)
            
                
            # prepare narrative mask
            narr_mask = torch.zeros(batch_merged_input["input_ids"].size())
            for k_id in range(int(batch_merged_input["input_ids"].size(0))):
                _narr_length = len(batch_input[k_id])
                narr_mask[k_id][:_narr_length+1] = 1  # include CLS

            narr_mask = narr_mask.to(device)

            optimizer.zero_grad()
            model.zero_grad()
            start_logits, end_logits = model(batch_merged_input)

            '''
            if len(batch_start_labels.size()) > 1:
                batch_start_labels = batch_start_labels.squeeze(-1)
            if len(batch_end_labels.size()) > 1:
                batch_end_labels = batch_end_labels.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            '''

            # get the mask of the narratives
            attn_map = narr_mask
            inverse_attn_map = torch.ones(attn_map.size()).to(device) - attn_map
            inverse_attn_map = -100 * inverse_attn_map
            start_logits = (start_logits * attn_map) + inverse_attn_map
            end_logits = (end_logits * attn_map) + inverse_attn_map



            loss_start = criterionSpan(start_logits, batch_start_labels)
            loss_end = criterionSpan(end_logits, batch_end_labels)

            
            loss = loss_start + loss_end

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_loss += loss.item() * len(batch_input)
            optimizer.step()
            del batch_input

        logging.info("Epoch {} avg loss = {}".format(epoch_for_train, float(epoch_loss)/float(len(all_inputs))) )
        
    

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
    all_test_pk = [] 

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
            for sq_key in span_map_single_example.keys():
                train_data.append((narr.lower().lower(), quote.lower(), sq_key, span_map_single_example[sq_key] ))


        elif pk in test_pk:
            assert pk not in train_pk
            for sq_key in span_map_single_example.keys():
                test_data.append((narr.lower().lower(), quote.lower(), sq_key, span_map_single_example[sq_key] ))
                all_test_pk.append(pk)
        else:
            ValueError


    shuffle(train_data)
    
    all_narratives_train = [x[0] for x in train_data]
    all_quotes_train = [x[1] for x in train_data]
    all_spans_train = [x[2] for x in train_data]
    all_spans_train_label =  [x[3] for x in train_data]  # span from narratives
    


    all_narratives_test = [x[0] for x in test_data]    
    all_quotes_test = [x[1] for x in test_data]
    all_spans_test =  [x[2] for x in test_data]
    all_spans_test_label =  [x[3] for x in test_data]
    

    qtok = tokenize_and_make_tensor(all_quotes, model_type)
    qtok = qtok.to(device)

        
    model = SpanModel(qtok, model_type)
    model = model.to(device)
    
    
    # NOTE: for test or start from previously saved checkpoint
    # checkpoint = torch.load("./path_to_model") 
    # model.load_state_dict(checkpoint["model"], strict=False)

    #NOTE: below code is for training
    model.train()
    model.set_train()

    param_optimizer = list(model.named_parameters())

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = hyper_params["learning_rate"], correct_bias=False)
    # optimizer.load_state_dict(checkpoint["opt"])
    
    ("Number of parameters to train: {}".format(int(np.sum(np.array([len(_x["params"]) for _x in optimizer_grouped_parameters ]))) ))
    
    
    debug = False
    
    prec_list = []
    recall_list = []

    best_prec_1 = -1
    best_recall_1 = -1
    best_epoch_1 = -1

    best_prec_2 = -1
    best_recall_2 = -1
    best_epoch_2 = -1
    

    # NOTE: uncoment below to test only
    # pred_labels, probs, all_probs = tester(model, model_type, all_narratives_test, all_spans_test, all_spans_test_label, labels_test_tensor, epoch_for_test=0, debug=True)
    
    
    
    # NOTE: code below is for training
    start_epoch_id = 0

    for i in range(start_epoch_id, hyper_params["num_epochs"] + start_epoch_id):
        
        logging.info("Epoch main {}".format(i))
        model, optimizer = trainer(model, model_type, optimizer, all_narratives_train,all_quotes_train, all_spans_train, all_spans_train_label, epoch_for_train=i)
        
        torch_random_state = torch.get_rng_state()
        recall, precision = tester(model, model_type, all_test_pk, all_narratives_test, all_quotes_test, all_spans_test, all_spans_test_label, epoch_for_test=0, debug=True)
        torch.set_rng_state(torch_random_state)

        if (recall > best_recall_1) or ( (recall == best_recall_1) and (precision > best_prec_1) ):
            best_recall_1 = recall
            best_prec_1 = precision
            best_epoch_1 = i
            save_model(model, optimizer, monitoring = "recall")

        if (precision > best_prec_2) or ( (precision == best_prec_2) and (recall > best_recall_2) ):
            best_recall_2 = recall
            best_prec_2 = precision
            best_epoch_2 = i
            save_model(model, optimizer, monitoring = "precision")




if __name__=="__main__":
    # take key from main args
    parser = argparse.ArgumentParser(description='span prediction')
    parser.add_argument('--model', type=str, choices = list(LLM_model.keys()), help='specify model name')    
    args = parser.parse_args()

    model_name = str(args.model).lower()
    if model_name not in list(LLM_model.keys()):
        raise ValueError("model not in list")

    log_filename = "./logs_span_prediction/{}_seen.log".format(model_name)

    logging.basicConfig(
        filename=log_filename, 
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Finetuning : {}".format(model_name))
    hyper_params["save_dir"] = "./saved_models_span_prediction/{}_seen/".format(model_name)
    main(model_name)
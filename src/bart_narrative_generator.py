from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, RobertaTokenizer,BertTokenizer
import json
import numpy as np
import random
from random import shuffle
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import os
import sys
from torchtext.data.metrics import bleu_score
from copy import deepcopy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW
from torch.utils.data import  RandomSampler, SequentialSampler
from torchtext.data.metrics import bleu_score

hyper_params = {
    "seed" : 123,
    "max_length_quote": 15,
    "max_length_narrative": 256,
    "batch_size": 8,
    "num_epochs": 10,
    "learning_rate": 1e-5,
    "train_test_split": 0.6,
    "dataset_file": "./data/full_dataset.json",
    "experiment_name" : "bart_narrative_generation",
    "save_dir" : "bart_narrative_generation_results" 
}

if not os.path.exists(hyper_params["save_dir"]):
    os.makedirs(hyper_params["save_dir"])


def shift_tokens_right(input_ids, pad_token_id):
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens



def encode_batch(source_sentences, target_sentences, pad_to_max_length=True, return_tensors="pt"):
    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}
    
    for sentence in source_sentences:
        # print(type(sentence))
        encoded_dict = tokenizer(
            sentence,
            max_length=hyper_params["max_length_narrative"],
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=hyper_params["max_length_quote"],
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
        )
        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)
    

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return batch


def train(all_narratives_train, labels_train):
    model.train()
    batch_size = hyper_params["batch_size"]
    num_batches = len(all_narratives_train) // batch_size
    if len(all_narratives_train) % batch_size != 0:
        num_batches += 1
    
    train_loss = 0
    for bidx in range(num_batches):
        if bidx != num_batches - 1:
            train_data_batch = all_narratives_train[bidx * batch_size : (bidx + 1)*batch_size]
            train_labels_batch = labels_train[bidx * batch_size : (bidx + 1)*batch_size]
        else:
            train_data_batch = all_narratives_train[bidx * batch_size :]
            train_labels_batch = labels_train[bidx * batch_size :]
        
        batch_tensors = encode_batch(train_data_batch, train_labels_batch)
        src_ids, src_mask, target_ids = batch_tensors["input_ids"], batch_tensors["attention_mask"], batch_tensors["labels"]
        decoder_input_ids = shift_tokens_right(target_ids, tokenizer.pad_token_id)

        src_ids = src_ids.to(device)
        src_mask = src_mask.to(device)
        target_ids = target_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)

        outputs = model(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        optimizer.zero_grad()
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        del batch_tensors
    
    
    print("Total train loss = ", train_loss) 



def get_ed_list(quote, list_of_quotes):
    quote = quote.lower().strip()
    quote = quote.split()
    ed_list = []
    for _qt in list_of_quotes:
        qt = _qt.lower().strip()
        qt = qt.split()

        n, m = len(quote), len(qt)
        if n > m:
            quote,qt = qt,quote
            n,m = m,n
            
        current = range(n+1)
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if quote[j-1] != qt[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)
                
        ed_list.append(current[n])

    return ed_list


def test(all_narratives_test, labels_test, list_of_quotes ):
    model.eval()
    batch_size = 4
    num_batches = len(all_narratives_test) // batch_size
    if len(all_narratives_test) % batch_size != 0:
        num_batches += 1
    
    test_bleu = 0
    test_acc = 0
    for bidx in range(num_batches):
        if bidx != num_batches - 1:
            test_data_batch = all_narratives_test[bidx * batch_size : (bidx + 1)*batch_size]
            test_labels_batch = labels_test[bidx * batch_size : (bidx + 1)*batch_size]
        else:
            test_data_batch = all_narratives_test[bidx * batch_size :]
            test_labels_batch = labels_test[bidx * batch_size :]
        
        batch_tensors = encode_batch(test_data_batch, test_labels_batch)
        src_ids, src_mask, target_ids = batch_tensors["input_ids"], batch_tensors["attention_mask"], batch_tensors["labels"]
        
        src_ids = src_ids.to(device)
        src_mask = src_mask.to(device)
        target_ids = target_ids.to(device)
        
        quote_ids = model.generate(src_ids, attention_mask=src_mask, num_beams=4, max_length=hyper_params["max_length_quote"], early_stopping=True, use_cache=True, decoder_start_token_id = tokenizer.pad_token_id)

        quote_gen = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in quote_ids]
        quote_gen = [q.strip() for q in quote_gen]

        
        for i in range(len(test_labels_batch)):
            generated_quotes = [quote_gen[i].split()]
            reference_quotes = [[test_labels_batch[i].split()]]
            test_bleu += bleu_score(generated_quotes, reference_quotes)


            ed_list = get_ed_list(quote_gen[i].lower(), list_of_quotes)
            min_ind = np.argmin(np.array(ed_list))

            retr_quote = list_of_quotes[min_ind]

            if retr_quote.lower().strip() == test_labels_batch[i].lower().strip():
                test_acc += 1

        del batch_tensors

    test_bleu = float(test_bleu) / float(len(all_narratives_test))
    test_acc = float(test_acc) / float(len(all_narratives_test))

    print("Avg bleu score = ", test_bleu)
    print("Avg acc = ", test_acc)

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


dataset = json.load(open("dataset_with_all_keywords.json", "r")) # NOTE:compute keywords using spacy beforehand 
# and create a file "dataset_with_all_keywords.json" same format as full dataset.
# Add one more key called "keywords" to the "fields" dictionary of each entry, which 
# contains list of all keywords

dataset_map = {}

for d in dataset:
    quote = d["fields"]["quote"].strip().strip("\n").lower()
    narr = d["fields"]["narrative"].strip().strip("\n").lower()

    keywords = d["fields"]["keywords"]
    keywords = [k.lower().strip().strip("\n") for k in keywords]

    t = (narr, keywords)

    if quote not in dataset_map.keys():
        dataset_map[quote] = []

    dataset_map[quote].append(t)



mode  = "seen" # or "unseen"

train_data = []
test_data = []

if mode == "seen":
    for c in dataset_map.keys():
        quote = c
        narr_key_all = dataset_map[c]
        shuffle(narr_key_all)
        num_train = int(0.8 * len(narr_key_all))
        train_narr_key_all = narr_key_all[:num_train]
        test_narr_key_all = narr_key_all[num_train:]
        
        for x in train_narr_key_all:
            train_data.append((quote, x[0], x[1] ))
        for x in test_narr_key_all:
            test_data.append((quote, x[0], x[1] ))
            
        

elif mode == "unseen" :
    all_quotes = list(dataset_map.keys())
    shuffle(all_quotes)

    train_narratives = []
    train_quotes = all_quotes[:200]
    test_quotes = all_quotes[200:]

    
    for c in dataset_map.keys():
        quote = c
        narr_key_all = dataset_map[c]

        if quote in train_quotes:
            for x in narr_key_all:
                train_data.append((quote, x[0], x[1] ))
                
                train_narratives.append(x[0])


        if quote in test_quotes:
            for x in narr_key_all:
                test_data.append((quote, x[0], x[1] ))

    
    
shuffle(train_data)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir="./cache/")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", cache_dir = "./cache/")
    
use_mode = "test"
if use_mode == "train":
    train_features = []
    max_len = 0
    max_len_narrative = 0
    max_seq_length = 141
    max_seq_length_narrative = 224
    pad_token_id = tokenizer.pad_token_id
    all_input_ids = []
    all_input_mask = []
    all_target_ids = []
    all_decoder_input_ids = []

    for i in range(len(train_data)):
        quote = train_data[i][0]
        narr = train_data[i][1]
        narrative_tokens = tokenizer.tokenize(narr)

        target_tokens = ["<s>"] + narrative_tokens + ["</s>"]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        if len(target_ids) > max_len_narrative:
            max_len_narrative = len(target_ids)

        keywords = train_data[i][2]

        quote_tokens = tokenizer.tokenize(quote)

        tokens = ["<s>"] + quote_tokens + ["</s>"]
        
        for k in keywords:
            keyword_tokens = tokenizer.tokenize(k)
            tokens += keyword_tokens + ["</s>"]

        decoder_input_tokens = ["</s>"] + target_tokens
        decoder_input_ids = tokenizer.convert_tokens_to_ids(decoder_input_tokens)

        if len(tokens) > max_len:
            max_len = len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [pad_token_id] * (max_seq_length - len(input_ids))
        input_ids += padding
        
        padding = [0] * (max_seq_length - len(input_mask))
        input_mask += padding
        


        padding = [pad_token_id] * (max_seq_length_narrative - len(decoder_input_ids))
        
        decoder_input_ids += padding

        padding = [pad_token_id] * (max_seq_length_narrative - len(target_ids))
        target_ids += padding
        

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_target_ids.append(target_ids)
        all_decoder_input_ids.append(decoder_input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(target_ids) == max_seq_length_narrative
        assert len(decoder_input_ids) == max_seq_length_narrative
        

    

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_target_ids = torch.tensor(all_target_ids, dtype=torch.long)
    all_decoder_input_ids = torch.tensor(all_decoder_input_ids, dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_decoder_input_ids, all_target_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=4)
            

    device = torch.device("cuda")

    param_optimizer = list(model.named_parameters())

    
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)


    model.to(device)

    for ep in range(15):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            batch_input_ids, batch_input_mask, batch_decoder_input_ids, batch_target_ids = batch
            
            optimizer.zero_grad()


            outputs = model(batch_input_ids, attention_mask=batch_input_mask, decoder_input_ids=batch_decoder_input_ids, use_cache=False)
            lm_logits = outputs[0]
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), batch_target_ids.view(-1))

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()


        print("Total training loss = ", tr_loss)
        torch.save({"model":model.state_dict(), "opt":optimizer.state_dict()} , "./bart_narrative_generator_model_{}.pth".format(ep))





else:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    test_features = []
    max_len = 0
    max_len_narrative = 0
    max_seq_length = 141  # seen
    max_seq_length_narrative = 250
    pad_token_id = tokenizer.pad_token_id

    
    all_input_ids = []
    all_input_mask = []
    all_target_ids = []
    all_decoder_input_ids = []

    all_quotes = []
    all_keywords = []
    all_true_narratives = []

    for i in range(len(test_data)):
        quote = test_data[i][0]
        narr = test_data[i][1]
        all_true_narratives.append(narr)
        
        keywords = test_data[i][2]
        all_quotes.append(quote)
        all_keywords.append(keywords)

        quote_tokens = tokenizer.tokenize(quote)

        tokens = ["<s>"] + quote_tokens + ["</s>"]
        
        for k in keywords:
            keyword_tokens = tokenizer.tokenize(k)
            tokens += keyword_tokens + ["</s>"]

        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(input_ids) > max_len:
            max_len = len(input_ids)

        input_mask = [1] * len(input_ids)

        padding = [pad_token_id] * (max_seq_length - len(input_ids))
        input_ids += padding
        padding = [0] * (max_seq_length - len(input_mask))
        input_mask += padding
        
        

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)

        assert len(input_ids) == max_seq_length 
        assert len(input_mask) == max_seq_length
        
    print(all_input_ids[0])
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)

            

    device = torch.device("cuda")
    model.to(device)
    checkpoint = torch.load("./new_model_bart_unseen_14.pth")
    model.load_state_dict(checkpoint["model"])
    model.eval()


    generated_narratives = []
    all_bleu = []
    all_rouge = []
    all_recall = []
    for i in trange(len(all_input_ids)):
        gen_sample = {}
        batch_input_ids = all_input_ids[i].unsqueeze(0).to(device)
        batch_input_mask =  all_input_mask[i].unsqueeze(0).to(device)
        
        narrative_ids = model.generate(batch_input_ids, attention_mask=batch_input_mask, num_beams=4, max_length=250, early_stopping=True, use_cache=True, decoder_start_token_id = tokenizer.pad_token_id)

        
        narrative_gen = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in narrative_ids]
        narrative_gen = [q.strip() for q in narrative_gen]
        narrative_gen = narrative_gen[0].strip().lower()


        true_narrative = all_true_narratives[i].strip("\n").strip().lower()

        cand = [narrative_gen.split()]
        ref = [[true_narrative.split()]]

        bleu = bleu_score(cand, ref)
        all_bleu.append(bleu)

        rouge = scorer.score(true_narrative,narrative_gen)
        rouge = rouge["rougeL"].fmeasure
        all_rouge.append(rouge)
        
        recall = 0
        for k in all_keywords[i]:
            if k in narrative_gen:
                recall += 1

        quote = tokenizer.decode(batch_input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        recall = recall / float(len(all_keywords[i]))
        all_recall.append(recall)
    
        
        gen_sample["quote"] = all_quotes[i]
        gen_sample["keywords"] = all_keywords[i]
        gen_sample["narrative_generated"] = narrative_gen
        gen_sample["true_narrative"] = true_narrative
        gen_sample["bleu"] = bleu
        gen_sample["rouge"] = rouge
        gen_sample["recall"] = recall
        generated_narratives.append(gen_sample)


    assert len(all_bleu) == len(generated_narratives)
    assert len(all_rouge) == len(generated_narratives)
    assert len(all_recall) == len(generated_narratives)

    print("Avg bleu = ", np.mean(np.array(all_bleu)))
    print("Avg rouge = ", np.mean(np.array(all_rouge)))
    print("Avg recall = ", np.mean(np.array(all_recall)))
    print("Num gen narratives = ", len(generated_narratives))
    
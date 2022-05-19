from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import numpy as np
import random
from random import shuffle
import torch
import torch.nn as nn
from tqdm import tqdm,trange
import os
import sys
from torchtext.data.metrics import bleu_score
from copy import deepcopy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, Adafactor
from torch.utils.data import  RandomSampler, SequentialSampler


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
    train_narratives = []

    for c in dataset_map.keys():
        quote = c
        narr_key_all = dataset_map[c]
        shuffle(narr_key_all)
        num_train = int(0.8 * len(narr_key_all))
        train_narr_key_all = narr_key_all[:num_train]
        test_narr_key_all = narr_key_all[num_train:]
        
        for x in train_narr_key_all:
            train_data.append((quote, x[0], x[1] ))
            train_narratives.append(x[0])

        for x in test_narr_key_all:
            test_data.append((quote, x[0], x[1] ))
            

if mode == "unseen" :
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

tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir="./cache/")
model = T5ForConditionalGeneration.from_pretrained("t5-base", cache_dir = "./cache/")


use_mode = "test"
if use_mode == "train":
    train_features = []
    max_len = 0
    max_len_narrative = 0
    max_seq_length = 153
    max_seq_length_narrative = 251
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




    optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True, lr=None)

    model.to(device)
    for ep in range(15):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            batch_input_ids, batch_input_mask, batch_decoder_input_ids, batch_target_ids = batch
            
            optimizer.zero_grad()


            outputs = model(batch_input_ids, attention_mask=batch_input_mask, decoder_input_ids=batch_decoder_input_ids,use_cache=False)
            lm_logits = outputs[0]
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), batch_target_ids.view(-1))

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()


        print("Total training loss = ", tr_loss)
        torch.save({"model":model.state_dict(), "opt":optimizer.state_dict()} , "./t5_narrative_generator_model_{}.pth".format(ep))



else:
    from torchtext.data.metrics import bleu_score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    test_features = []
    max_len = 0
    max_len_narrative = 0
    max_seq_length = 153
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
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [pad_token_id] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        


        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)

            

    device = torch.device("cuda")
    model.to(device)
    checkpoint = torch.load("./load_path.pth") #NOTE:give model path
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

        quote = tokenizer.decode(batch_input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        rouge = scorer.score(true_narrative,narrative_gen)
        rouge = rouge["rougeL"].fmeasure
        
        all_rouge.append(rouge)
        recall = 0
        for k in all_keywords[i]:
            if k in narrative_gen:
                recall += 1
        
        
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
    print("Avg bleu = ", np.mean(np.array(all_bleu)))
    print("Avg rouge = ", np.mean(np.array(all_rouge)))
    print("Avg recall = ", np.mean(np.array(all_recall)))
    print("Num gen narratives = ", len(generated_narratives))
    
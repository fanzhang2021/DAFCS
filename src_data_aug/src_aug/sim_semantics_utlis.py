import time

import faiss
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm.auto import tqdm

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

MAX_LENGTH = 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CodesDataset(Dataset):
    def __init__(self, codes):
        self.codes = codes

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        return self.codes[i]

def sents_to_vecs(sents, tokenizer, model):
    dataset = CodesDataset(sents)
    dataLoader = DataLoader(dataset, 64, shuffle=False)

    vecs = []

    progress_bar_in = tqdm(range(len(dataLoader)))
    for codes in dataLoader:
        with torch.no_grad():
            batch_tokenized = tokenizer(codes, add_special_tokens=True,
                                             padding=True, max_length=MAX_LENGTH,
                                             truncation=True, return_tensors="pt")  # tokenize、add special token、pad

            input_ids = batch_tokenized['input_ids'].to(DEVICE)
            attention_mask = batch_tokenized['attention_mask'].to(DEVICE)

            with autocast():
                hidden_states = model(input_ids, attention_mask=attention_mask, return_dict=True,
                                       output_hidden_states=True).hidden_states

            output_hidden_state = hidden_states[-1]
            outputs = output_hidden_state[:, 0, :]

            vec = outputs.cpu().numpy()

            vecs.extend(vec)

            progress_bar_in.update(1)


    assert len(sents) == len(vecs)
    vecs = np.array(vecs)

    return vecs  # shape: all_num * 768



def bulid_index(vecs):
    df_text = pd.DataFrame(vecs).astype('float32')

    df_text = np.ascontiguousarray(np.array(df_text))

    dim, measure = 768, faiss.METRIC_L2
    param = 'HNSW64'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)
    index.add(df_text)

    return index

def find_nearst_samples(query, index, topk):
    D, I = index.search(query, topk)
    print("text near examples, topk is ", topk)
    print(I)
    print(D)

def bulid_index_cos_sim(vecs):
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    return index

def find_nearst_cos(query, index, topk):
    faiss.normalize_L2(query)
    D, I = index.search(query, topk)
    return I




if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained("../CODEBERT")
    model = RobertaModel.from_pretrained("../CODEBERT")
    model.to(DEVICE)

    sents = ["123","assasas","348","assasas","12"]
    vecs = sents_to_vecs(sents, tokenizer, model)


    query = ["123"]
    query = sents_to_vecs(query, tokenizer, model)
    print("len(query)", len(query))

    index = bulid_index_cos_sim(vecs)

    topk_index = find_nearst_cos(query, index, 2)

    print(topk_index)




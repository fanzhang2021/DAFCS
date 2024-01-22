from transformers import RobertaTokenizer, RobertaModel
import faiss
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm.auto import tqdm

from src_aug.sim_lexical_utlis import sim_jaccard, sim_jaccard_fliter_keywords, sim_syntactic_fliter_keywords
from src_aug.sim_semantics_utlis import sents_to_vecs, bulid_index_cos_sim, find_nearst_cos
from src_aug.read_data import read_source_data, read_target_data

def write_source_data_to_file(out_file_path, all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels):
    with open(out_file_path, "w") as writer:
        for code_sta, code_org, query, label in zip(all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels):
            code_sta = str(code_sta).replace("'", '"')
            writer.write('<CODESPLIT>'.join([str(code_sta), code_org, query, label]) + '\n')

def write_source_data_to_file_for_train(out_file_path, all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels):
    with open(out_file_path, "w") as writer:
        for code_sta, code_org, query, label in zip(all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels):
            code_sta = str(code_sta).replace("'", '"')
            writer.write(str(label) + '<CODESPLIT>'+ "URL<CODESPLIT>" + '<CODESPLIT>'.join([str(code_sta), query, code_org]) + '\n')

def Nlist_to_1_list(all_t_satements_code):
    all = []
    for sta_list in all_t_satements_code:
        all.extend(sta_list)

    return all


def get_sim_statement(query, tokenizer, model, all_statements_index, topk, all_statements, lang, lexical_threshold=0.3):
    query_vec = sents_to_vecs([query], tokenizer, model)
    i_th_topk_index = find_nearst_cos(query_vec, all_statements_index, topk)
    ith_index = i_th_topk_index[0]

    all_sim_statements = []
    for i in ith_index:
        sim_statement = all_statements[i]

        code_score = sim_syntactic_fliter_keywords(query, sim_statement, lang)
        if (code_score > lexical_threshold and query != sim_statement):
            all_sim_statements.append(sim_statement)

    return all_sim_statements

def bulid_statements_index(all_statements, tokenizer, model):
    all_statements_vecs = sents_to_vecs(list(all_statements), tokenizer, model)
    print("len(target_codes_vecs)", len(all_statements_vecs))

    # index
    all_statements_index = bulid_index_cos_sim(all_statements_vecs)

    return all_statements_index
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

def main():
    topk = 10
    lexical_threshold = 0
    lang = 'sql'

    train_num = 100


    out_file_path = "../data_out/"+str(train_num)+"_shared_query_codes.txt"

    source_file_path = "../data_out/sql_source_statements_codes.txt"
    all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels = read_source_data(source_file_path, splitnum=train_num)

    target_file_path = "../data_out/sql_target_statements_codes.txt"
    all_t_satements_code, all_t_parse_codes = read_target_data(target_file_path)

    tokenizer = RobertaTokenizer.from_pretrained("../CODEBERT")
    model = RobertaModel.from_pretrained("../CODEBERT")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    source_codes_vecs = sents_to_vecs(list(all_s_parse_codes), tokenizer, model)
    target_codes_vecs = sents_to_vecs(list(all_t_parse_codes), tokenizer, model)
    print("len(source_codes_vecs)", len(source_codes_vecs))
    print("len(target_codes_vecs)", len(target_codes_vecs))

    target_index = bulid_index_cos_sim(target_codes_vecs)

    all_topk_index = []
    for q_vec in source_codes_vecs:
        q_vec = np.array([q_vec])
        i_th_topk_index = find_nearst_cos(q_vec, target_index, topk)
        all_topk_index.append(i_th_topk_index[0])

    assert (len(all_topk_index) == len(source_codes_vecs))

    all_queries, all_codes, all_lables, all_statements = [], [], [], []
    for i_th_topk_index, s_satements, source_code, source_query, source_label \
            in zip(all_topk_index, all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels):
        sim_codes = []
        sim_codes_satements = []
        for j in i_th_topk_index:
            target_code = all_t_parse_codes[j]
            code_score = sim_jaccard_fliter_keywords(target_code, source_code, lang)
            if (code_score > lexical_threshold):
                sim_codes.append(target_code)
                sim_codes_satements.append(all_t_satements_code[j])

        print(source_query)
        print(source_code)
        print(sim_codes)
        print()

        for t_code, t_statements in zip(sim_codes, sim_codes_satements):
            all_queries.append(source_query)
            all_codes.append(t_code)
            all_lables.append(source_label)
            all_statements.append(t_statements)

    write_source_data_to_file_for_train(out_file_path, all_statements, all_codes, all_queries, all_lables)



if __name__ == '__main__':
    main()
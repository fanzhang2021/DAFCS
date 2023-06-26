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
            code_sta = str(code_sta).replace("'", '"')  # 不这样处理，会使得后续分割为list出错，但是这样处理，会影响代码的"'"
            writer.write('<CODESPLIT>'.join([str(code_sta), code_org, query, label]) + '\n')

def write_source_data_to_file_for_train(out_file_path, all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels):
    with open(out_file_path, "w") as writer:
        for code_sta, code_org, query, label in zip(all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels):
            code_sta = str(code_sta).replace("'", '"')  # 不这样处理，会使得后续分割为list出错，但是这样处理，会影响代码的"'"
            writer.write(str(label) + '<CODESPLIT>'+ "URL<CODESPLIT>" + '<CODESPLIT>'.join([str(code_sta), query, code_org]) + '\n')

def Nlist_to_1_list(all_t_satements_code):
    #将list嵌套list，转换为一个list
    all = []
    for sta_list in all_t_satements_code:
        all.extend(sta_list)

    return all


def get_sim_statement(query, tokenizer, model, all_statements_index, topk, all_statements, lang, lexical_threshold=0.3):
    # 语义相似 在tall_statements_index中找topk相似的
    query_vec = sents_to_vecs([query], tokenizer, model)
    i_th_topk_index = find_nearst_cos(query_vec, all_statements_index, topk)
    ith_index = i_th_topk_index[0]

    # print("ith_index: ", ith_index)

    all_sim_statements = []
    for i in ith_index:
        sim_statement = all_statements[i]

        code_score = sim_syntactic_fliter_keywords(query, sim_statement, lang)
        if (code_score > lexical_threshold and query != sim_statement):
            # print(code_score)
            # print(sim_statement)
            all_sim_statements.append(sim_statement)

    return all_sim_statements

def bulid_statements_index(all_statements, tokenizer, model):
    #先语义相似，再取文本相似

    # 转化为向量
    all_statements_vecs = sents_to_vecs(list(all_statements), tokenizer, model)
    print("len(target_codes_vecs)", len(all_statements_vecs))

    # 建立索引
    all_statements_index = bulid_index_cos_sim(all_statements_vecs)

    return all_statements_index


def test():
    #先语义相似，再取文本相似
    topk = 1000
    lexical_threshold = 0
    lang = 'sql'

    # 读取文件
    target_file_path = "../data_out/sql_target_statements_codes.txt"
    all_t_satements_code, all_t_parse_codes = read_target_data(target_file_path, splitnum=0)

    all_statements = Nlist_to_1_list(all_t_satements_code)

    # 语义模型
    tokenizer = RobertaTokenizer.from_pretrained("../CODEBERT")
    model = RobertaModel.from_pretrained("../CODEBERT")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    # 转化为向量
    all_statements_vecs = sents_to_vecs(list(all_statements), tokenizer, model)
    print("len(target_codes_vecs)", len(all_statements_vecs))

    # 建立索引
    all_statements_index = bulid_index_cos_sim(all_statements_vecs)

    query = "FROM table_name_8"
    get_sim_statement(query, tokenizer, model, all_statements_index, topk, all_statements, lang, lexical_threshold)





if __name__ == '__main__':
    test()
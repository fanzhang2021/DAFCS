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

def main():
    #先语义相似，再取文本相似
    topk = 10
    lexical_threshold = 0
    lang = 'sql'

    train_num = 100


    # 增强后的输出文件
    out_file_path = "../data_out/"+str(train_num)+"_shared_query_codes.txt"

    # 读取文件
    source_file_path = "../data_out/sql_source_statements_codes.txt"
    all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels = read_source_data(source_file_path, splitnum=train_num)

    target_file_path = "../data_out/sql_target_statements_codes.txt"
    all_t_satements_code, all_t_parse_codes = read_target_data(target_file_path)

    # 语义模型
    tokenizer = RobertaTokenizer.from_pretrained("../CODEBERT")
    model = RobertaModel.from_pretrained("../CODEBERT")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    # 转化为向量
    source_codes_vecs = sents_to_vecs(list(all_s_parse_codes), tokenizer, model)
    target_codes_vecs = sents_to_vecs(list(all_t_parse_codes), tokenizer, model)
    print("len(source_codes_vecs)", len(source_codes_vecs))
    print("len(target_codes_vecs)", len(target_codes_vecs))

    # 建立索引
    target_index = bulid_index_cos_sim(target_codes_vecs)

    # 语义相似
    # source code中的每个语句，在target_code中找topk相似的
    all_topk_index = []
    for q_vec in source_codes_vecs:
        q_vec = np.array([q_vec])
        i_th_topk_index = find_nearst_cos(q_vec, target_index, topk)
        all_topk_index.append(i_th_topk_index[0]) #存放query最相似的下标 list中套着list,形如[a,b,c,f,e]的target_index是[[1,2,3], [1,7,8], [2,5,6]···]
        # print(topk_index[0])

    assert (len(all_topk_index) == len(source_codes_vecs))

    #找到all_topk_index对应的原始code
    all_queries, all_codes, all_lables, all_statements = [], [], [], []
    for i_th_topk_index, s_satements, source_code, source_query, source_label \
            in zip(all_topk_index, all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels):
        sim_codes = []
        sim_codes_satements = [] #target code对应的语句
        for j in i_th_topk_index: #将第i个source code，对应的相似的code索引一个一个的拿出来，然后找到对应的原始code
            target_code = all_t_parse_codes[j]
            # 语义相似 和 文本相似 的 交集
            code_score = sim_jaccard_fliter_keywords(target_code, source_code, lang)
            if (code_score > lexical_threshold):
                sim_codes.append(target_code)
                sim_codes_satements.append(all_t_satements_code[j])

        print(source_query)
        print(source_code)
        print(sim_codes)
        print()

        #共用query，label
        for t_code, t_statements in zip(sim_codes, sim_codes_satements):
            all_queries.append(source_query)
            all_codes.append(t_code)
            all_lables.append(source_label)
            all_statements.append(t_statements) #这里append的是target code的语句

        # 将原始的也再次添加上，一起写入文件
        # all_queries.append(source_query)
        # all_codes.append(source_code)
        # all_lables.append(source_label)
        # all_statements.append(s_satements)


    # print(all_queries)
    # print(all_codes)
    # print(all_lables)
    # print(all_statements)

    # 写入文件
    # write_source_data_to_file(out_file_path, all_statements, all_codes, all_queries, all_lables)
    write_source_data_to_file_for_train(out_file_path, all_statements, all_codes, all_queries, all_lables)



if __name__ == '__main__':
    main()
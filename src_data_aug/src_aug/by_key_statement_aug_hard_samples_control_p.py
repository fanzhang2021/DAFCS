import copy
import math
import random

from transformers import RobertaTokenizer, RobertaModel
import faiss
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm.auto import tqdm

from src_aug.find_sim_statements import bulid_statements_index, get_sim_statement, Nlist_to_1_list
from src_aug.sim_lexical_utlis import sim_jaccard, sim_jaccard_fliter_keywords, sim_syntactic_fliter_keywords
from src_aug.sim_semantics_utlis import sents_to_vecs, bulid_index_cos_sim, find_nearst_cos
from src_aug.read_data import read_source_data, read_target_data, read_source_data_with_key_statements

def write_aug_2_file(out_file_path, labels, text_lines, code_lines, reserve_key_stas, reserve_origin_codes):
    with open(out_file_path, "a", encoding="utf-8") as writer:
        for label, aug_query, aug_code, key_sta, orin_code in zip(labels, text_lines, code_lines, reserve_key_stas, reserve_origin_codes):
            writer.write(str(label) + '<CODESPLIT>'+ '<CODESPLIT>'.join(
                [key_sta, orin_code, aug_query, aug_code]) + '\n')

def dedup(all_sim_statements):
    fliter_list = []
    for s in all_sim_statements:
        if(s not in fliter_list):
            fliter_list.append(s)
    return fliter_list

def solidity_fuc_declar_statements(all_t_statements):
    #取第一句，就能拿到函数申明语句
    all_declar = []
    for t_stas in all_t_statements:
        all_declar.append(t_stas[0])

    return all_declar

def select_st(origin_st, candidate_st_list, all_t_statements):
    r_list = []
    for st in candidate_st_list:
        if(origin_st != st):
            r_list.append(st)
            # return st
        if (len(r_list) == 10): #从topk 10中随机选一个，更多样一些
            break
    if(len(r_list) != 0):
        return random.choice(r_list)
    else:
        print("len(candidate_st_list) {}   len(r_list) {}".format(len(candidate_st_list), len(r_list)))
        return random.choice(all_t_statements) #如果r_list为空，无法选择，则随机用一个

def select_st_num(origin_st, candidate_st_list, num):
    r_list = []
    for st in candidate_st_list:
        if(origin_st != st):
            r_list.append(st)
        if(len(r_list) == num*3):
            break
    if(len(r_list) < num):
        print("len(candidate_st_list) {}   len(r_list) {}, num{}".format(len(candidate_st_list), len(r_list), num))
    min_num = min(num, len(r_list))
    return random.sample(r_list, min_num) #加点随机性，更多样一些


def get_random_unSim_st(st, all_t_statements, lang):
    while (True):
        random_st = random.choice(all_t_statements)
        code_score = sim_syntactic_fliter_keywords(st, random_st, lang) #通过文本不匹配找到不相似的语句
        if(code_score <= 0.2):
            return random_st

    return random_st


def aug_insert(lang, source_file_path, tokenizer, model, all_target_statements_index, all_t_statements, all_declar, all_declar_index, pos_p, neg_p):
    err_num = 0
    # source_file_path = "../data_out/100_key_stas.txt"
    all_source_labels, all_source_queries, all_source_codes, all_key_stas, all_trival_stas, all_source_statements = read_source_data_with_key_statements(
        source_file_path)

    # 开始aug
    # 插入 插入重要语句的相似语句，至任意位置
    aug_labels, aug_queries, aug_codes = [], [], []
    reserve_key_stas, reserve_origin_codes = [], []
    for l, q, c, key_st, trival_st, code_statements in zip(all_source_labels, all_source_queries, all_source_codes,
                                                           all_key_stas, all_trival_stas, all_source_statements):
        if(int(l) == 1):
            # 寻找到相似的语句
            all_sim_statements = get_sim_statement(key_st, tokenizer, model, all_target_statements_index, topk=1000,
                                                   all_statements=all_t_statements, lang=lang)
            all_sim_statements = dedup(all_sim_statements)  # 去重
            # all_sim_statements = all_sim_statements[:8]  # 截取
            aug_num = math.ceil(pos_p * len(code_statements))
            all_sim_statements = select_st_num(key_st, all_sim_statements, aug_num)

            # 插入任意位置
            new_code_statements = copy.deepcopy(code_statements)
            for sim_st in all_sim_statements:
                # pos = random.randint(0, len(code_statements))
                pos = random.randint(1, min(len(code_statements), 5))
                # pos = 1
                new_code_statements.insert(pos, sim_st)

                # print(" ".join(new_code_statements))

            # 增强后存放
            aug_labels.append(int(l))
            aug_queries.append(q)
            aug_codes.append(" ".join(new_code_statements))

            reserve_key_stas.append(key_st)
            reserve_origin_codes.append(c)

            # ######### 增强相对于正例的困难反例   ########
            # #要不先删除关键语句？再插入反义。   插入的反义语句数据量过多，会导致反样本长，正样本短，模型可能会学错
            # aug_num_delete = math.ceil(neg_p * len(code_statements))
            # # 随机选择要删除的语句
            # selected_code_statements = random.sample(code_statements, k=aug_num_delete)
            # if (key_st not in selected_code_statements):  # 把重要的句子放进去
            #     selected_code_statements.append(key_st)
            #
            # new_code_statements = copy.deepcopy(code_statements)
            #
            # for st in selected_code_statements:
            #     try:
            #         new_code_statements.remove(st)
            #     except Exception as e:
            #         err_num += 1
            #
            # #插入语义相反的语句
            # aug_num_insert = math.ceil(neg_p * len(code_statements))
            # for st in selected_code_statements:
            #     un_sim_statements = get_random_unSim_st(st, all_t_statements, lang)
            #     # all_sim_statements = get_sim_statement(st, tokenizer, model, all_target_statements_index,
            #     #                                        topk=50000,
            #     #                                        all_statements=all_t_statements, lang=lang)
            #     # all_sim_statements = dedup(all_sim_statements)  # 去重
            #     # un_sim_statements = all_sim_statements[-1]  # 截取
            #     pos = 0
            #     new_code_statements.insert(pos, un_sim_statements)
            #
            # # 增强后存放
            # aug_labels.append(0)
            # aug_queries.append(q)
            # aug_codes.append(" ".join(new_code_statements))
            #
            # reserve_key_stas.append(key_st)
            # reserve_origin_codes.append(c)

            ######### 增强相对于正例的困难反例   ########
            # 组合模式  最重要的语句 组合 其他语句进行替换
            # 设置增强比例
            # p = 0.5
            aug_num = math.ceil(neg_p * len(code_statements))
            # 随机选择要替换的语句
            selected_code_statements = random.sample(code_statements, k=aug_num)
            if (key_st not in selected_code_statements):  # 把重要的句子放进去
                selected_code_statements.append(key_st)
            new_code_statements = copy.deepcopy(code_statements)

            # 替换
            for st in selected_code_statements:
                # random_sta_1 = random.choice(all_t_statements)
                if (lang == "solidity" and "function" in st):  # 如果是solidity语言，那么重要语句的替换语句，在函数申明中选择
                    un_sim_declar = get_random_unSim_st(st, all_declar, lang)
                    # all_sim_declar = get_sim_statement(st, tokenizer, model, all_declar_index,
                    #                                        topk=50000,
                    #                                        all_statements=all_declar, lang=lang)
                    # all_sim_declar  = dedup(all_sim_declar)  # 去重
                    # un_sim_declar = all_sim_declar[-1]  # 截取
                    random_sta_1 = un_sim_declar

                else:
                    un_sim_statements = get_random_unSim_st(st, all_t_statements, lang)
                    # all_sim_statements = get_sim_statement(st, tokenizer, model, all_target_statements_index,
                    #                                        topk=50000,
                    #                                        all_statements=all_t_statements, lang=lang)
                    # all_sim_statements = dedup(all_sim_statements)  # 去重
                    # un_sim_statements = all_sim_statements[-1]  # 截取
                    random_sta_1 = un_sim_statements
                try:
                    pos = new_code_statements.index(st)
                    new_code_statements[pos] = random_sta_1
                except Exception as e:
                    err_num += 1
            # 增强后存放
            if (len(new_code_statements) != 0):
                aug_labels.append(0)
                aug_queries.append(q)
                aug_codes.append(" ".join(new_code_statements))

                reserve_key_stas.append(key_st)
                reserve_origin_codes.append(c)


    return aug_labels, aug_queries, aug_codes, reserve_key_stas, reserve_origin_codes


def aug_replace(lang, source_file_path, tokenizer, model, all_target_statements_index, all_t_statements, all_declar, all_declar_index, pos_p, neg_p):
    err_num = 0
    # source_file_path = "../data_out/100_key_stas.txt"
    all_source_labels, all_source_queries, all_source_codes, all_key_stas, all_trival_stas, all_source_statements = read_source_data_with_key_statements(
        source_file_path)

    # 开始aug
    # 替换 将不重要的语句，替换为其相似的语句  #TODO 从demo上看，要不要替换最不重要的语句更好？  或者可以考虑，每个语句都替换 N 次，除了最重要的
    aug_labels, aug_queries, aug_codes = [], [], []
    reserve_key_stas, reserve_origin_codes = [], []
    for l, q, c, key_st, trival_st, code_statements in zip(all_source_labels, all_source_queries, all_source_codes,
                                                           all_key_stas, all_trival_stas, all_source_statements):

        if(int(l) == 1):
            # 增强正例
            # 设置增强比例
            # p = 0.5
            #随机选择要替换的语句
            copy_code_statements = copy.deepcopy(code_statements)
            if(key_st in copy_code_statements): #偶尔出现意外
                copy_code_statements.remove(key_st)#因为主要句子不替换
            aug_num = math.ceil(pos_p * len(copy_code_statements))
            selected_code_statements = random.sample(copy_code_statements, k=aug_num)
            new_code_statements = copy.deepcopy(code_statements)
            for st in selected_code_statements:
                if (st != key_st):
                    not_importent_st = st
                    # 寻找到相似的语句
                    all_sim_statements = get_sim_statement(not_importent_st, tokenizer, model,
                                                           all_target_statements_index, topk=1000,
                                                           all_statements=all_t_statements, lang=lang)
                    all_sim_statements = dedup(all_sim_statements)  # 去重
                    sim_st = select_st(st, all_sim_statements, all_t_statements)

                    try:
                        pos = new_code_statements.index(not_importent_st)
                        new_code_statements[pos] = sim_st
                        # print("relpace new_code_statements ", new_code_statements)
                        # print()
                    except Exception as e:
                        err_num += 1
            # 增强后存放
            if (len(new_code_statements) != 0 and new_code_statements != code_statements):
                aug_labels.append(int(l))
                aug_queries.append(q)
                aug_codes.append(" ".join(new_code_statements))
                reserve_key_stas.append(key_st)
                reserve_origin_codes.append(c)



            ######### 增强相对于正例的困难反例   ########
            #组合模式  最重要的语句 组合 其他语句进行替换
            # 设置增强比例
            # p = 0.5
            aug_num = math.ceil(neg_p * len(code_statements))
            # 随机选择要替换的语句
            selected_code_statements = random.sample(code_statements, k=aug_num)
            if(key_st not in selected_code_statements):#把重要的句子放进去
                selected_code_statements.append(key_st)
            new_code_statements = copy.deepcopy(code_statements)

            #替换
            for st in selected_code_statements:
                # random_sta_1 = random.choice(all_t_statements)
                if (lang == "solidity" and "function" in st):  # 如果是solidity语言，那么重要语句的替换语句，在函数申明中选择
                    un_sim_declar = get_random_unSim_st(st, all_declar, lang)
                    # all_sim_declar = get_sim_statement(st, tokenizer, model, all_declar_index,
                    #                                        topk=50000,
                    #                                        all_statements=all_declar, lang=lang)
                    # all_sim_declar  = dedup(all_sim_declar)  # 去重
                    # un_sim_declar = all_sim_declar[-1]  # 截取
                    random_sta_1 = un_sim_declar

                else:
                    un_sim_statements = get_random_unSim_st(st, all_t_statements, lang)
                    # all_sim_statements = get_sim_statement(st, tokenizer, model, all_target_statements_index,
                    #                                        topk=50000,
                    #                                        all_statements=all_t_statements, lang=lang)
                    # all_sim_statements = dedup(all_sim_statements)  # 去重
                    # un_sim_statements = all_sim_statements[-1]  # 截取
                    random_sta_1 = un_sim_statements
                try:
                    pos = new_code_statements.index(st)
                    new_code_statements[pos] = random_sta_1
                except Exception as e:
                    err_num += 1
            # 增强后存放
            if (len(new_code_statements) != 0):
                aug_labels.append(0)
                aug_queries.append(q)
                aug_codes.append(" ".join(new_code_statements))

                reserve_key_stas.append(key_st)
                reserve_origin_codes.append(c)


    return err_num, aug_labels, aug_queries, aug_codes, reserve_key_stas, reserve_origin_codes

# 删除 删除不重要的语句 或者是 除了重要的语句，其余的都可以删除
def aug_delete(lang, source_file_path, all_t_statements, tokenizer, model, all_target_statements_index, all_declar, all_declar_index, pos_p, neg_p):
    err_num = 0
    # source_file_path = "../data_out/100_key_stas.txt"
    all_source_labels, all_source_queries, all_source_codes, all_key_stas, all_trival_stas, all_source_statements = read_source_data_with_key_statements(
        source_file_path)

    # 开始aug
    # 删除不重要的语句
    aug_labels, aug_queries, aug_codes = [], [], []
    reserve_key_stas, reserve_origin_codes = [], []
    for l, q, c, key_st, trival_st, code_statements in zip(all_source_labels, all_source_queries, all_source_codes,
                                                           all_key_stas, all_trival_stas, all_source_statements):
        if(int(l) == 1):
            # 除了最重要的，别的按比例删除
            # p = 0.5
            aug_num = math.ceil(pos_p * len(code_statements))
            # 随机选择要替换的语句
            selected_code_statements = random.sample(code_statements, k=aug_num)
            if (key_st in selected_code_statements):  # 排除最重要的句子
                selected_code_statements.remove(key_st)

            new_code_statements = copy.deepcopy(code_statements)
            for st in selected_code_statements:
                try:
                    new_code_statements.remove(st)
                except Exception as e:
                    err_num += 1

            # 增强后存放
            if (len(new_code_statements) != 0):
                aug_labels.append(int(l))
                aug_queries.append(q)
                aug_codes.append(" ".join(new_code_statements))

                reserve_key_stas.append(key_st)
                reserve_origin_codes.append(c)

            # ######### 增强相对于正例的困难反例   ########
            # # 要不先删除关键语句？再插入反义。   插入的反义语句数据量过多，会导致反样本长，正样本短，模型可能会学错
            # aug_num_delete = math.ceil(neg_p * len(code_statements))
            # # 随机选择要删除的语句
            # selected_code_statements = random.sample(code_statements, k=aug_num_delete)
            # if (key_st not in selected_code_statements):  # 把重要的句子放进去
            #     selected_code_statements.append(key_st)
            #
            # new_code_statements = copy.deepcopy(code_statements)
            #
            # for st in selected_code_statements:
            #     try:
            #         new_code_statements.remove(st)
            #     except Exception as e:
            #         err_num += 1
            #
            # # 插入语义相反的语句
            # aug_num_insert = math.ceil(neg_p * len(code_statements))
            # for st in selected_code_statements:
            #     un_sim_statements = get_random_unSim_st(st, all_t_statements, lang)
            #     # all_sim_statements = get_sim_statement(st, tokenizer, model, all_target_statements_index,
            #     #                                        topk=50000,
            #     #                                        all_statements=all_t_statements, lang=lang)
            #     # all_sim_statements = dedup(all_sim_statements)  # 去重
            #     # un_sim_statements = all_sim_statements[-10]  # 截取
            #     pos = 0
            #     new_code_statements.insert(pos, un_sim_statements)
            #
            # # 增强后存放
            # aug_labels.append(0)
            # aug_queries.append(q)
            # aug_codes.append(" ".join(new_code_statements))
            #
            # reserve_key_stas.append(key_st)
            # reserve_origin_codes.append(c)

            ######### 增强相对于正例的困难反例   ########
            # 组合模式  最重要的语句 组合 其他语句进行替换
            # 设置增强比例
            # p = 0.5
            aug_num = math.ceil(neg_p * len(code_statements))
            # 随机选择要替换的语句
            selected_code_statements = random.sample(code_statements, k=aug_num)
            if (key_st not in selected_code_statements):  # 把重要的句子放进去
                selected_code_statements.append(key_st)
            new_code_statements = copy.deepcopy(code_statements)

            # 替换
            for st in selected_code_statements:
                # random_sta_1 = random.choice(all_t_statements)
                if (lang == "solidity" and "function" in st):  # 如果是solidity语言，那么重要语句的替换语句，在函数申明中选择
                    un_sim_declar = get_random_unSim_st(st, all_declar, lang)
                    # all_sim_declar = get_sim_statement(st, tokenizer, model, all_declar_index,
                    #                                        topk=50000,
                    #                                        all_statements=all_declar, lang=lang)
                    # all_sim_declar  = dedup(all_sim_declar)  # 去重
                    # un_sim_declar = all_sim_declar[-1]  # 截取
                    random_sta_1 = un_sim_declar

                else:
                    un_sim_statements = get_random_unSim_st(st, all_t_statements, lang)
                    # all_sim_statements = get_sim_statement(st, tokenizer, model, all_target_statements_index,
                    #                                        topk=50000,
                    #                                        all_statements=all_t_statements, lang=lang)
                    # all_sim_statements = dedup(all_sim_statements)  # 去重
                    # un_sim_statements = all_sim_statements[-1]  # 截取
                    random_sta_1 = un_sim_statements
                try:
                    pos = new_code_statements.index(st)
                    new_code_statements[pos] = random_sta_1
                except Exception as e:
                    err_num += 1
            # 增强后存放
            if (len(new_code_statements) != 0):
                aug_labels.append(0)
                aug_queries.append(q)
                aug_codes.append(" ".join(new_code_statements))

                reserve_key_stas.append(key_st)
                reserve_origin_codes.append(c)

    return err_num, aug_labels, aug_queries, aug_codes, reserve_key_stas, reserve_origin_codes

def add_contract_head(solidity_aug_codes):
    head_solidity_aug_codes = []

    print(len(solidity_aug_codes))

    for code in solidity_aug_codes:
        code = code.replace("function","") #为了让正例和反例长得相似，防止模型学错重点
        code = code.strip()
        id = random.randint(10000, 100000)
        head = "contract c" + str(id) + "{ function "
        head_code = head + code + " }"
        head_solidity_aug_codes.append(head_code)

    assert (len(head_solidity_aug_codes) == len(solidity_aug_codes))

    return head_solidity_aug_codes

def assert_equal(aug_labels, aug_queries, aug_codes, reserve_key_stas, reserve_origin_codes):
    print("len {}, {}, {}, {}, {}".format(len(aug_labels), len(aug_queries), len(aug_codes), len(reserve_key_stas), len(reserve_origin_codes)))
    assert (len(aug_labels) == len(aug_queries))
    assert (len(aug_queries) == len(aug_codes))
    assert ((len(aug_labels) + len(aug_queries) + len(aug_codes) + len(reserve_key_stas) + len(reserve_origin_codes)) / 5 == len(aug_labels))



def aug_for_hard(lang, source_file_path, target_file_path, query_aug_model, train_num):
    # lang = 'sql'
    # out_file_path = "../data_out/100_aug.txt"

    # 读取源文件
    # source_file_path = "../data_out/1000_key_stas.txt"
    # all_source_labels, all_source_queries, all_source_codes, all_key_stas, all_trival_stas, all_source_statements = read_source_data_with_key_statements(source_file_path)

    print(read_source_data_with_key_statements(source_file_path, 2))

    # 读取目标文件
    # target_file_path = "../data_out/sql_target_statements_codes.txt"
    all_t_satements_code, all_t_parse_codes = read_target_data(target_file_path, splitnum=train_num)
    all_t_statements = Nlist_to_1_list(all_t_satements_code)

    #函数声明语句，只用于solidity
    all_declar = solidity_fuc_declar_statements(all_t_satements_code)

    # 语义模型
    tokenizer = RobertaTokenizer.from_pretrained("../CODEBERT")
    model = RobertaModel.from_pretrained("../CODEBERT")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    # 对目标文件建立索引
    all_target_statements_index = bulid_statements_index(all_t_statements, tokenizer, model)
    all_declar_index = bulid_statements_index(all_declar, tokenizer, model)

    for p in [0.15, 0.25, 0.35, 0.45]:  # 由易到难
        out_file_path = "../data_out/" + lang + "/" + str(train_num)+ "/" + str(train_num) + "_aug_" + str(p) + ".txt"
        pos_p = p  # 正例 插入、删除、替换的越少，样本越简单
        neg_p = 1.15 - p  # 反例 删除、替换的越多，样本越简单

        # 开始 code aug
        insert_aug_labels, insert_aug_queries, insert_aug_codes, insert_reserve_key_stas, insert_reserve_origin_codes \
            = aug_insert(lang, source_file_path, tokenizer, model, all_target_statements_index, all_t_statements, all_declar, all_declar_index, pos_p, neg_p)
        err_num_replace, replace_aug_labels, replace_aug_queries, replace_aug_codes, replace_reserve_key_stas, replace_reserve_origin_codes \
            = aug_replace(lang, source_file_path, tokenizer, model, all_target_statements_index, all_t_statements, all_declar, all_declar_index, pos_p, neg_p)
        err_num_delete, delete_aug_labels, delete_aug_queries, delete_aug_codes, delete_reserve_key_stas, delete_reserve_origin_codes = aug_delete(
            lang, source_file_path, all_t_statements, tokenizer, model, all_target_statements_index, all_declar, all_declar_index, pos_p, neg_p)

        # 增强 queries
        # insert_aug_queries = query_aug_model.augment(insert_aug_queries)
        # replace_aug_queries = query_aug_model.augment(replace_aug_queries)
        # delete_aug_queries = query_aug_model.augment(delete_aug_queries)

        #如果是solidity，添加head
        if(lang == "solidity"):
            insert_aug_codes = add_contract_head(insert_aug_codes)
            replace_aug_codes = add_contract_head(replace_aug_codes)
            delete_aug_codes = add_contract_head(delete_aug_codes)

        #写入前判断
        assert_equal(insert_aug_labels, insert_aug_queries, insert_aug_codes, insert_reserve_key_stas, insert_reserve_origin_codes)
        assert_equal(replace_aug_labels, replace_aug_queries, replace_aug_codes, replace_reserve_key_stas, replace_reserve_origin_codes)
        assert_equal(delete_aug_labels, delete_aug_queries, delete_aug_codes, delete_reserve_key_stas, delete_reserve_origin_codes)

        # 追加写入文件
        write_aug_2_file(out_file_path, insert_aug_labels, insert_aug_queries, insert_aug_codes, insert_reserve_key_stas,
                         insert_reserve_origin_codes)
        write_aug_2_file(out_file_path, replace_aug_labels, replace_aug_queries, replace_aug_codes,
                         replace_reserve_key_stas, replace_reserve_origin_codes)
        write_aug_2_file(out_file_path, delete_aug_labels, delete_aug_queries, delete_aug_codes, delete_reserve_key_stas,
                         delete_reserve_origin_codes)

        print("err_num_replace {}, err_num_delete {}".format(err_num_replace, err_num_delete))

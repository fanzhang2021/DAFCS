import pprint

import Levenshtein
from src_aug.read_data import *
# from src_aug.second_find_sim_lexical_codes import sim_syntactic

def sim_syntactic(ast1, ast2):
    return Levenshtein.seqratio(str(ast1).split(), str(ast2).split())

def open_keyword_file(keyword_file_path):
    with open(keyword_file_path, encoding="utf-8") as f:
        all_keywords = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    return all_keywords

def fliter_keywords_func(s1, all_keywords):
    fliter_s1 = []
    for s in s1:
        if (s not in all_keywords):
            fliter_s1.append(s)

    return fliter_s1

def sim_jaccard_fliter_keywords(s1, s2, lang=""):
    s1 = s1.split()
    s2 = s2.split()

    keyword_file_path = "../data_in/" + lang + "_keywords"
    all_keywords = open_keyword_file(keyword_file_path)

    s1 = fliter_keywords_func(s1, all_keywords)
    s2 = fliter_keywords_func(s2, all_keywords)

    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

def sim_jaccard(s1, s2):
    s1 = s1.split()
    s2 = s2.split()

    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

def sim_lexical(code1, code2):
    return sim_jaccard(code1, code2)

def sim_syntactic_fliter_keywords(s1, s2, lang):
    s1 = s1.split()
    s2 = s2.split()

    keyword_file_path = "../data_in/" + lang + "_keywords"
    all_keywords = open_keyword_file(keyword_file_path)

    s1 = fliter_keywords_func(s1, all_keywords)
    s2 = fliter_keywords_func(s2, all_keywords)
    return Levenshtein.seqratio(s1, s2)


def get_sim_target_codes(source_code, target_code_list, sim_criterion, sim_range):
    if(sim_criterion != "lexical" and sim_criterion != "syntactic"):
        print("err sim_criterion: ", sim_criterion)
        return

    sim_floor = float(sim_range[0])
    sim_cell = float(sim_range[1])
    # 处理文本相似度
    sim_target_index = []
    target_index = 0
    for target_code in target_code_list:
        if (sim_criterion == "lexical"):
            sim_value = sim_lexical(source_code, target_code)
        elif(sim_criterion == "syntactic"):
            sim_value = sim_syntactic(source_code, target_code)

        if (sim_floor < sim_value <= sim_cell):
            sim_target_index.append(target_index) #实际保存的是target_code_list对应的下标

            print(source_code)
            print(target_code)
            print(sim_value)

            print()

        target_index += 1

    return sim_target_index

#获取对source_code满足相似度要求的所有target_code 的下标
def get_sourceCode_2_targetCode_index(source_code_list, target_code_list, sim_criterion, sim_range):
    """

    :param source_code_list: 所有的源代码: list
    :param target_code_list: 所有的目标代码: list
    :param sim_criterion: 相似度评价方式
    :param sim_range: 相似度值的范围
    :return: 对于在source_code_list中的每个code，返回符合相似度要求的target code list；一起返回: list中套着list,形如[a,b,c,f,e]的target_index是[[1,2,3], [1], [2,5,6]···]
    """
    all_sim_target_code_index = []  # 所有满足source code相似度要求的 target_index，都存放在这里，形如[a,b,c,f,e]的target_index是[[1,2,3], [1], [2,5,6]···]
    for s_code in source_code_list:
        sim_target_code_index = get_sim_target_codes(s_code, target_code_list, sim_criterion=sim_criterion,
                                                     sim_range=sim_range)

        all_sim_target_code_index.append(
            sim_target_code_index)  # 即使sim_target_code_index返回的是[],也会被添加里面去占位，这是合理的，否则后续处理顺序会乱
        # print(all_sim_target_code_index) #实时观察

    # pprint.pp(all_sim_target_code_index)
    # print(len(all_sim_target_code_index))
    assert (len(all_sim_target_code_index) == len(source_code_list))
    return all_sim_target_code_index

def test_get_sim_target_codes():
    source_code = "SELECT T2.Comptroller FROM election AS T1 JOIN party AS T2 ON T1.Party"

    sim_range = [0.6, 1]  # 相似度设置为区间
    file_path = "../data_out/sql_templates_codes.txt"
    all_coarse_templates, all_middle_templates, all_fine_templates, all_satements_code, all_parse_codes = read_target_data(
        file_path)

    sim_target_code = get_sim_target_codes(source_code, all_parse_codes, sim_criterion="lexical",
                                           sim_range=sim_range)  # lexical syntactic

    print(len(sim_target_code))

if __name__ == '__main__':

    sim_range = [0.5, 1]
    sim_criterion = "lexical"

    source_file_path = "../data_out/sql_source_statements_codes.txt"
    all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels = read_source_data(source_file_path)

    target_file_path = "../data_out/sql_target_statements_codes.txt"
    all_t_satements_code, all_t_parse_codes = read_target_data(target_file_path)

    #相似度区间过滤
    sourceCode_2_targetCode_index = get_sourceCode_2_targetCode_index(all_s_parse_codes, all_t_parse_codes, sim_criterion, sim_range)

    print(sourceCode_2_targetCode_index)
    # print(len(sourceCode_2_targetCode_index))









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

    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

def sim_jaccard(s1, s2):
    s1 = s1.split()
    s2 = s2.split()

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

    sim_target_index = []
    target_index = 0
    for target_code in target_code_list:
        if (sim_criterion == "lexical"):
            sim_value = sim_lexical(source_code, target_code)
        elif(sim_criterion == "syntactic"):
            sim_value = sim_syntactic(source_code, target_code)

        if (sim_floor < sim_value <= sim_cell):
            sim_target_index.append(target_index)

            print(source_code)
            print(target_code)
            print(sim_value)

            print()

        target_index += 1

    return sim_target_index

def get_sourceCode_2_targetCode_index(source_code_list, target_code_list, sim_criterion, sim_range):

    all_sim_target_code_index = []
    for s_code in source_code_list:
        sim_target_code_index = get_sim_target_codes(s_code, target_code_list, sim_criterion=sim_criterion,
                                                     sim_range=sim_range)

        all_sim_target_code_index.append(
            sim_target_code_index)
    assert (len(all_sim_target_code_index) == len(source_code_list))
    return all_sim_target_code_index

def test_get_sim_target_codes():
    source_code = "SELECT T2.Comptroller FROM election AS T1 JOIN party AS T2 ON T1.Party"

    sim_range = [0.6, 1]
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

    sourceCode_2_targetCode_index = get_sourceCode_2_targetCode_index(all_s_parse_codes, all_t_parse_codes, sim_criterion, sim_range)

    print(sourceCode_2_targetCode_index)
    # print(len(sourceCode_2_targetCode_index))









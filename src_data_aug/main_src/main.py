import random
import numpy as np
import torch
import nlpaug.augmenter.word as naw
from src_aug.by_key_statement_aug_hard_samples_control_p import aug_for_hard
from src_saliency.find_key_statement_all_label import find_key_statements

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_aug(train_num, lang):

    #读入文件配置
    source_file_path = "../data_out/"+ lang + "_source_statements_codes.txt" #读取的source code，已经经过分语句的文件
    target_file_path = "../data_out/"+ lang + "_target_statements_codes.txt" #读取的target code, 已经经过分语句的文件

    #输出文件配置
    out_file_path_for_find_key_statements = "../data_out/"+ lang + "/" + str(train_num) + "_key_stas.txt"
    # out_file_path_for_aug = "../data_out/"+ lang + "/" + str(train_num) + "_aug.txt"

    # 语义模型
    T_model_teacher = "../save_model/teacher/teacher_python_model.pkl"
    T_model = torch.load(T_model_teacher)

    # query_aug_model = naw.ContextualWordEmbsAug(
    #     model_path='bert-base-uncased', action="insert", aug_p=0.15, device='cuda')
    query_aug_model = None

    # 1 确定code片段中的key语句，并保存到文件中
    find_key_statements(train_num, source_file_path, out_file_path_for_find_key_statements, T_model)

    # 2 增强
    # aug(lang, out_file_path_for_aug, out_file_path_for_find_key_statements, target_file_path, query_aug_model)
    aug_for_hard(lang, out_file_path_for_find_key_statements, target_file_path, query_aug_model,
                 train_num)

    #之后便可以train了



if __name__ == '__main__':
    set_seed(1)
    lang = "sql"
    for train_num in [100, 500, 1000, 5000, 8000, 10000, 14000]:
        run_aug(train_num, lang)

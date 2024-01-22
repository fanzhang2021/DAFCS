import numpy as np
import torch
from apex.amp import scaler
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from src_aug.read_data import read_source_data
from src_saliency.Module_Util import Teacher_BertClassfication, Teacher_Dataset_For_Lable, Removed_Dataset_For_Lable, remove_sta, \
    remove_a_sta


def write_key_data_to_file(out_file_path, all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels, all_key_stas, all_trival_stas):
    with open(out_file_path, "w") as writer:
        for code_sta, code_org, query, label, key_sta, trival_sta in zip(all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels, all_key_stas, all_trival_stas):
            code_sta = str(code_sta).replace("'", '"')
            writer.write(str(label) + '<CODESPLIT>'+ "URL<CODESPLIT>" + '<CODESPLIT>'.join([str(code_sta), query, code_org, key_sta, trival_sta]) + '\n')

def meatures_a_code_remove_statement(satements_code, code, query, label, T_model, position_label):

    # 逐个去除语句的 removed_dataLoader
    code, ith_remove_sta_codes = remove_a_sta(satements_code, code)

    removed_dataset = Removed_Dataset_For_Lable(query, ith_remove_sta_codes, label)
    removed_dataLoader = DataLoader(removed_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    all_remove_sta_logits = []
    for text, code, label in removed_dataLoader:
        print(list(text))
        print(list(code))
        targets = torch.tensor(list(label)).to(device)
        with torch.no_grad():
            with autocast():
                outputs = T_model(list(text), list(code))
                predict_propety = outputs[:, position_label]

        all_remove_sta_logits.extend(predict_propety.detach().cpu().numpy())

    key_index = np.argmin(all_remove_sta_logits)
    trival_index = np.argmax(all_remove_sta_logits)

    key_sta = satements_code[key_index]
    trival_sta = satements_code[trival_index]

    print("position_label:", position_label)
    print("ith_remove_sta_codes", ith_remove_sta_codes)
    print("all_remove_sta_logits:", all_remove_sta_logits)
    print("key_index sta:", satements_code[key_index])
    print("trival_index sta:", satements_code[trival_index])

    print()

    return key_sta, trival_sta


def meature_all_codes(all_satements_code, all_codes, all_queries, all_labels,T_model):
    all_key_stas, all_trival_stas = [], []
    for q, c, l, stas in zip(all_queries, all_codes, all_labels, all_satements_code):
        if (int(l) == 1):
            key_sta, trival_sta = meatures_a_code_remove_statement(stas, c, q, l, T_model, position_label = 1)
        else:
            key_sta, trival_sta = meatures_a_code_remove_statement(stas, c, q, l, T_model, position_label = 0)

        all_key_stas.append(key_sta)
        all_trival_stas.append(trival_sta)

    return all_key_stas, all_trival_stas


def find_key_statements(train_num, source_file_path, out_file_path, T_model):

    all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels = read_source_data(source_file_path,
                                                                                            splitnum=train_num)

    all_key_stas, all_trival_stas = meature_all_codes(all_s_satements_code, all_s_parse_codes, all_s_queries,
                                                      all_s_labels, T_model)

    write_key_data_to_file(out_file_path, all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels,
                           all_key_stas, all_trival_stas)

if __name__ == '__main__':

    train_num = 1000
    out_file_path = "../data_out/"+str(train_num)+"_key_stas.txt"

    source_file_path = "../data_out/sql_source_statements_codes.txt"
    all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels = read_source_data(source_file_path,
                                                                                            splitnum=train_num)
    # code, all_remove_sta_codes = remove_a_sta(all_s_satements_code[0], all_s_parse_codes[0])
    # print(code)
    # print(all_remove_sta_codes)
    #
    T_model_teacher = "../save_model/teacher/teacher_python_model.pkl"
    T_model = torch.load(T_model_teacher)

    all_key_stas, all_trival_stas = meature_all_codes(all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels, T_model)

    write_key_data_to_file(out_file_path, all_s_satements_code, all_s_parse_codes, all_s_queries, all_s_labels, all_key_stas, all_trival_stas)
















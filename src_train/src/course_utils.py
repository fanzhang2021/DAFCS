import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from src.inference import main_inference
from src.mrr import get_mrr


class Train_Dataset(Dataset):
    def __init__(self, queries, codes, labels):
        self.text_lines = queries
        self.code_lines = codes
        self.labels = labels

        print("Aug_Dataset 训练的数据量:", len(labels))

    def __len__(self):
        return len(self.text_lines)  # 注意这个len本质是数据的数量

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, int(c)


def read_aug_data_with_propety(point_aug_data_path):
    print("point_aug_data_path: ", point_aug_data_path)
    labels, queries, codes, all_p_propety= [], [], [], []

    with open(point_aug_data_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # max_range = min_range + delta

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        p_loss = float(temp_line[3])
        p_propety = float(temp_line[4])
        p_label = int(temp_line[5])

        # if (p_label == int(temp_line[0])): #加这一层过滤吗？要求teacher预测的标签，和增强前预定义的标签相同。
        if (p_propety >= 0.85):
            labels.append(int(temp_line[0]))
            queries.append(temp_line[1])
            codes.append(temp_line[2])
            all_p_propety.append(p_propety)

    assert (len(labels) == len(queries) == len(codes) == len(all_p_propety))

    return labels, queries, codes, all_p_propety


def read_aug_data_for_postive(point_aug_data_path, hard_start, hard_end):
    hard_start = hard_start - 0.05
    hard_end = hard_end - 0.05
    print("point_aug_data_path: ", point_aug_data_path)
    labels, queries, codes= [], [], []

    with open(point_aug_data_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # max_range = min_range + delta

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        p_loss = float(temp_line[3])
        p_propety = float(temp_line[4])
        p_label = int(temp_line[5])
        # if(p_propety >= min_range and p_propety <= max_range):
        if (1 == int(temp_line[0])):
            if (p_propety >= hard_start and p_propety < hard_end):
                if(p_label == int(temp_line[0])):
                    labels.append(int(temp_line[0]))
                    queries.append(temp_line[1])
                    codes.append(temp_line[2])

    # print("read_aug_data_for_postive, read file name:{}, 符合要求大于 {} 的比例:{}/{}".format(point_aug_data_path, min_range, len(labels), len(lines)))
    print("read file name:{}, 符合range {}-{} 的比例:{}/{}".format(point_aug_data_path, hard_start, hard_end, len(labels), len(lines)))

    return labels, queries, codes


def read_aug_data_for_negtive(point_aug_data_path, hard_start, hard_end):
    print("point_aug_data_path: ", point_aug_data_path)
    labels, queries, codes= [], [], []

    with open(point_aug_data_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # max_range = min_range + delta

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        p_loss = float(temp_line[3])
        p_propety = float(temp_line[4])
        p_label = int(temp_line[5])
        # if(p_propety >= min_range and p_propety <= max_range):
        if (0 == int(temp_line[0])):
            if (p_propety >= hard_start and p_propety < hard_end):
                labels.append(int(temp_line[0]))
                queries.append(temp_line[1])
                codes.append(temp_line[2])

    print("read_aug_data_for_negtive, read file name:{}, 符合range {}-{} 的比例:{}/{}".format(point_aug_data_path, hard_start, hard_end, len(labels), len(lines)))

    return labels, queries, codes


class Aug_Dataset_for_course(Dataset):
    def __init__(self, queries, codes, labels):
        self.text_lines = queries
        self.code_lines = codes
        self.labels = labels

        print("Aug_Dataset 训练的数据量:", len(labels))

    def __len__(self):
        return len(self.text_lines)  # 注意这个len本质是数据的数量

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, int(c)


def rank_by_propty(rank_factor, all_queries_codes):
    # topk排序
    zipped = zip(rank_factor, all_queries_codes)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]), reverse=True) #reverse=True 从大到小排序
    result = zip(*sort_zipped)
    x_axis, y_axis = [list(x) for x in result]

    # 处理回来，把[]拆开
    new_all_quries = []
    new_all_codes = []
    new_all_labels = []
    for yi in y_axis:
        new_all_quries.append(yi[0])
        new_all_codes.append(yi[1])
        new_all_labels.append(yi[2])

    return new_all_labels, new_all_quries, new_all_codes

def read_data_dirctly(file_path):
    print("aug_data_path: ", file_path)
    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    text_lines = []
    code_lines = []
    labels = []

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
            # if(str(temp_line[0]) == "1"): #1表示代码和注释对应着，0表示每对应
            text_lines.append(temp_line[-2])  # 注释
            code_lines.append(temp_line[-1])  # 代码
            labels.append(int(temp_line[0]))

    print("注释和代码总行数:", len(text_lines), len(code_lines))

    return labels, text_lines, code_lines

import os

import copy
import random

from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import torch

from src.course_utils import Aug_Dataset_for_course, read_data_dirctly
from src.my_util import LineByLineTextDataset, train_a_epoch, valid_acc, valid_mrr

from tqdm import tqdm
#############################################################
PROJECT_ROOT = os.getcwd()[:-4]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def merge_old_new(old_labels, old_queries, old_codes, new_labels_1, new_queries_1, new_codes_1, new_labels_0, new_queries_0, new_codes_0):

    merge_labels, merge_queries, merge_codes = [], [], []

    merge_labels.extend(old_labels)
    merge_queries.extend(old_queries)
    merge_codes.extend(old_codes)

    merge_labels.extend(new_labels_1)
    merge_queries.extend(new_queries_1)
    merge_codes.extend(new_codes_1)

    merge_labels.extend(new_labels_0)
    merge_queries.extend(new_queries_0)
    merge_codes.extend(new_codes_0)

    print("合并后的数据量 {}".format(len(merge_labels)))

    return merge_labels, merge_queries, merge_codes


def merge_data_3(augmented_file_path, hard_level, old_train_dataset):
    # 读取数据
    augmented_file_path = augmented_file_path.format(hard_level)
    labels, queries, codes = read_data_dirctly(augmented_file_path)

    print("第 {} hard level, 使用的数据量为 {}".format(hard_level,len(labels)))

    #合并之前的dataset
    # 和old query和code进行合并
    old_labels, old_queries, old_codes = old_train_dataset.labels, old_train_dataset.text_lines, old_train_dataset.code_lines
    merge_labels, merge_queries, merge_codes = \
        merge_old_new(old_labels, old_queries, old_codes, labels, queries, codes, [], [], [])

    return merge_labels, merge_queries, merge_codes


def run_train(lr, train_num):
    set_seed(42)
    # 加载最好的model
    from my_util import BertClassfication
    # best_model = torch.load(model_save_dir)
    best_model = BertClassfication(device, PROJECT_ROOT)
    best_model.to(device)
    writer = None

    #dataloader
    train_dataset = LineByLineTextDataset(file_path=train_file_path, train_num=train_num)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True)

    valid_dataset = LineByLineTextDataset(file_path=valid_file_path, train_num=0)
    valid_dataLoader = DataLoader(valid_dataset, 16, shuffle=False)

    ######################### train #########################
    best_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, best_model.parameters()), lr=lr)
    scheduler = None
    scaler = GradScaler()
    lossfuction = nn.CrossEntropyLoss()

    inner_epoch = 1
    max_accuracy, max_mrr = 0, 0
    best_model.train()
    progress_bar_2 = tqdm(range(len(train_dataLoader) * max_num_epochs))
    old_train_dataset = copy.deepcopy(train_dataset)
    for hard_level in [0.15, 0.25, 0.35, 0.45, 'end']:
        for epoch in range(max_num_epochs):
            # train
            best_model = train_a_epoch(best_model, train_dataLoader, lossfuction, scaler, best_optimizer, scheduler, writer,
                                       inner_epoch)
            # valid
            if(valid_model == "mrr"):
                current_mrr = valid_mrr(best_model, infer_file_path, output_infer_file, inner_epoch, writer, lang)
                print('current_mrr %.8f, max_mrr %.8f' % (current_mrr, max_mrr))
                if (current_mrr > max_mrr):
                    model_save_dir = PROJECT_ROOT + "/save_model/" + lang + "/model" + str(train_num) + "_" + str(lr) + ".pkl"
                    torch.save(best_model, model_save_dir)
                    max_model = copy.deepcopy(best_model)
                    max_mrr = current_mrr
                    print('max mrr %.8f' % (max_mrr))

                currnt_accuracy = valid_acc(best_model, valid_dataLoader, device, inner_epoch, writer)
                print('currnt_accuracy %.8f, max_accuracy %.8f' % (currnt_accuracy, max_accuracy))

            ##################################
            inner_epoch += 1
            progress_bar_2.update(1)
            ######################### train end#########################

        if(hard_level == 'end'):
            break

        #add data
        merge_labels, merge_queries, merge_codes = merge_data_3(augmented_file_path, hard_level, old_train_dataset)
        train_dataset = Aug_Dataset_for_course(queries=merge_queries, codes=merge_codes, labels=merge_labels)
        train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True)

        #early stop
        best_model = copy.deepcopy(max_model)
        best_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, best_model.parameters()), lr=lr)
        scaler = GradScaler()


if __name__ == '__main__':
    set_seed(42)
    ############################## 全局配置 ##############################
    T_grow = 6
    max_num_epochs = 10  # 每一个level的训练epoch
    batch_size = 64  # arg2
    lang = 'sql'

    valid_model = "mrr"

    # train path
    train_file_path = PROJECT_ROOT + "/data/train_valid/" + lang + "/train.txt"
    # infer path
    infer_file_path = PROJECT_ROOT + "/data/test/" + lang + "/batch_0.txt"
    output_infer_file = PROJECT_ROOT + "/results/" + lang + "/result_batch_0.txt"
    # valid
    valid_file_path = PROJECT_ROOT + "/data/train_valid/" + lang + "/valid.txt"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for train_num in [100, 500, 1000]: #train_num
        augmented_file_path = PROJECT_ROOT + "/data/train_valid/" + lang + "/" + str(train_num) + "/" + str(train_num) + "_aug_{}" + ".txt"

        for i_lr in [5e-5, 1e-5]: #tune lr
            lr = i_lr
            run_train(lr, train_num)



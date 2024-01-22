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

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str, train_num):
        print("read data file at:", file_path)
        assert os.path.isfile(file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        if(train_num != 0):
            self.lines = self.lines[:train_num]

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:
                self.text_lines.append(temp_line[-2])
                self.code_lines.append(temp_line[-1])
                self.labels.append(int(temp_line[0]))

        print("注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c


def train_a_epoch(model, train_dataLoader, lossfuction, scaler, optimizer, scheduler, writer, inner_epoch):

    epoch_all_loss = 0
    preds_label = []
    out_label_ids = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # progress_bar_in = tqdm(range(len(train_dataLoader) * max_num_epochs))
    model.train()
    for text, code, labels in train_dataLoader:
        # model.train()
        targets = labels.to(device)

        with autocast():
            outputs = model(list(text), list(code))
            loss = lossfuction(outputs, targets)

        scaler.scale(loss).backward()
        epoch_all_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        #scheduler.step() # not use scheduler
        optimizer.zero_grad()

        _, predict = torch.max(outputs, 1)
        preds_label.extend(predict.detach().cpu().numpy())
        out_label_ids.extend(targets.detach().cpu().numpy())

        # progress_bar_in.update(1)

    train_result = acc_and_f1(np.array(preds_label), np.array(out_label_ids))

    # every 5 epoch print loss
    current_loss = epoch_all_loss / len(train_dataLoader)
    print("PRE5 - inner_epoch: %d, loss: %.8f" % (inner_epoch + 1, current_loss))

    return model

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def valid_acc(model, valid_dataLoader, device, inner_epoch, writer):
    ########valid########
    model.eval()
    preds_label = []
    out_label_ids = []

    for text, code, labels in valid_dataLoader:
        # inputs = tuple(zip(text, code))
        label_list = labels.to(device)
        # with torch.no_grad():
        with autocast():
            outputs = model(list(text), list(code))

        _, predict = torch.max(outputs, 1)

        preds_label.extend(predict.detach().cpu().numpy())
        out_label_ids.extend(label_list.detach().cpu().numpy())

    valid_result = acc_and_f1(np.array(preds_label), np.array(out_label_ids))

    print("epoch: ", inner_epoch)
    currnt_accuracy = valid_result['acc']

    return currnt_accuracy

class BertClassfication(nn.Module):
    def __init__(self, device, my_root):
        super(BertClassfication, self).__init__()

        tokenizer_name = 'microsoft/graphcodebert-base'
        fine_tuned_model_path = my_root+"/fine_turn_GraBert"
        self.model = AutoModel.from_pretrained(fine_tuned_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.fc1 = nn.Linear(768, 2)  # 加的感知机做分类
        self.device = device

    def forward(self, text, code):  # 这里的输入是一个list
        batch_tokenized = self.tokenizer(list(text), list(code), add_special_tokens=True,
                                    padding=True, max_length=180,
                                    truncation=True, return_tensors="pt")  # tokenize、add special token、pad

        input_ids = batch_tokenized['input_ids'].to(self.device)

        attention_mask = batch_tokenized['attention_mask'].to(self.device)

        hidden_states = self.model(input_ids, attention_mask=attention_mask, return_dict=True,
                                   output_hidden_states=True).hidden_states

        output_hidden_state = hidden_states[-1]
        outputs = output_hidden_state[:, 0, :]
        output = self.fc1(outputs)

        return output


def valid_mrr(model, infer_file_path, output_infer_file, inner_epoch, writer, lang):
    #######valid########
    # 每个epoch都验证一下
    model.eval()
    main_inference(model, infer_file_path, output_infer_file, split_num=50000)
    current_mrr = get_mrr(lang)

    print("inner_epoch: ", inner_epoch)

    return current_mrr


def read_aug_data(file_path, splitnum=0):
    print("file_path: ", file_path)
    labels, queries, codes = [], [], []

    with open(file_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    #截断
    if(splitnum != 0):
        lines = lines[:splitnum]

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        if(len(temp_line)) == 5:
            labels.append(int(temp_line[0]))
            queries.append(temp_line[-2])
            codes.append(temp_line[-1])

    print("read file name:{}, num:{}".format(file_path, len(labels)))

    return labels, queries, codes

class Teacher_Dataset_For_Lable(Dataset):
    def __init__(self, queries, codes, labels, need_label):
        self.queries = queries
        self.codes = codes
        self.ls = labels
        self.need_label = need_label

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for q, c, l in zip(self.queries, self.codes, self.ls):
            if(self.need_label == int(l)):
                self.text_lines.append(q)
                self.code_lines.append(c)
                self.labels.append(l)

        print("read label {} aug data num {}".format(need_label, len(self.labels)))
    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = int(self.labels[i])
        return a, b, c


def rank_loss(all_loss, all_queries_codes):
    # topk排序
    zipped = zip(all_loss, all_queries_codes)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]), reverse=False)
    result = zip(*sort_zipped)
    x_axis, y_axis = [list(x) for x in result]

    # 处理回来，把[]拆开
    new_all_quries = []
    new_all_codes = []
    new_all_labels = []
    new_all_pre = []
    new_all_lo = []
    new_all_pl = []
    for yi in y_axis:
        new_all_quries.append(yi[0])
        new_all_codes.append(yi[1])
        new_all_labels.append(yi[2])
        new_all_pre.append(yi[3])
        new_all_lo.append(yi[4])
        new_all_pl.append(yi[5])

    return new_all_quries, new_all_codes, new_all_labels, new_all_pre, new_all_lo, new_all_pl


def write_aug_data_with_point(out_file_path, all_need_queries, all_need_codes, all_need_lables, all_current_loss, all_predict_propety, all_predic_label):
    with open(out_file_path, "a", encoding="utf-8") as writer:
        for text, code, label, p_loss, p_propety, p_label in zip(all_need_queries, all_need_codes, all_need_lables, all_current_loss, all_predict_propety, all_predic_label):
            writer.write(str(int(label)) + '<CODESPLIT>' + '<CODESPLIT>'.join(
                [text, code, str(p_loss), str(p_propety), str(p_label)]) + '\n')


def read_aug_data_with_point(point_aug_data_path, min_range, delta):
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
            if (p_propety >= min_range):
                if(p_label == int(temp_line[0])):
                    labels.append(int(temp_line[0]))
                    queries.append(temp_line[1])
                    codes.append(temp_line[2])
        else:
            if (p_label == int(temp_line[0])):
                labels.append(int(temp_line[0]))
                queries.append(temp_line[1])
                codes.append(temp_line[2])
            else:
                if (p_propety <= 0.35): #不是很有信心将label判断为1的，也可以当作0去用
                    labels.append(int(temp_line[0]))
                    queries.append(temp_line[1])
                    codes.append(temp_line[2])

    return labels, queries, codes


def merge_origin_aug(point_out_file_path, train_file_path, train_num, pr_1, pr_0): # pr_0 delta
    # load aug data
    labels_1, queries_1, codes_1 = read_aug_data_with_point(point_out_file_path.format(1), pr_1, pr_0)
    labels_0, queries_0, codes_0 = read_aug_data_with_point(point_out_file_path.format(0), pr_1, pr_0)

    balance_num = min(len(labels_1), len(labels_0))
    balance_labels_1, balance_queries_1, balance_codes_1 = labels_1[:balance_num], queries_1[:balance_num], codes_1[:balance_num]
    balance_labels_0, balance_queries_0, balance_codes_0 = labels_0[:balance_num], queries_0[:balance_num], codes_0[:balance_num]


    # origin data
    train_dataset = LineByLineTextDataset(file_path=train_file_path, train_num=train_num)
    origin_labels, origin_queries, origin_codes = train_dataset.labels, train_dataset.text_lines, train_dataset.code_lines

    # merge
    l = balance_labels_1 + balance_labels_0 + origin_labels
    q = balance_queries_1 + balance_queries_0 + origin_queries
    c = balance_codes_1 + balance_codes_0 + origin_codes

    return l, q, c

class Aug_Dataset(Dataset):
    def __init__(self, queries, codes, labels):
        self.text_lines = queries
        self.code_lines = codes
        self.labels = labels

        print("Aug_Dataset nums:", len(labels))

    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, int(c)
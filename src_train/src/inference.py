import torch
import os
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast as autocast

import transformers

from src.mrr import get_mrr

transformers.logging.set_verbosity_error()

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str, split_num):
        assert os.path.isfile(file_path)
        print("read data file at:", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断测试
        if(split_num != 0):
            self.lines = self.lines[:split_num]

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:
                self.text_lines.append(temp_line[-2]) #query
                self.code_lines.append(temp_line[-1]) #code
                self.labels.append(int(temp_line[0]))


        print("注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c


class BertClassfication(nn.Module):
    def __init__(self, device, my_root):
        super(BertClassfication, self).__init__()

        tokenizer_name = 'microsoft/codebert-base'
        self.model = AutoModel.from_pretrained(tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.fc1 = nn.Linear(768, 2)
        self.device = device

    def forward(self, text, code):
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



def write_result_to_file(output_test_file, all_result, test_data_dir, test_num):
    assert os.path.isfile(test_data_dir)
    print("read test file at:", test_data_dir)

    with open(test_data_dir, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    assert (len(lines) % test_num == 0)

    with open(output_test_file, "w") as writer:
        print("***** Output test results *****")
        for i, logit in tqdm(enumerate(all_result), desc='Testing'):
            # instance_rep = '<CODESPLIT>'.join(
            #     [item.encode('ascii', 'ignore').decode('ascii') for item in lines[i]])
            writer.write(lines[i] + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')


def main_inference(model, infer_file_path, output_infer_file, split_num):
    print("run inference")
    #config
    batch_size = 32

    ########################## Data #########################
    infer_dataset = LineByLineTextDataset(file_path=infer_file_path, split_num = split_num)
    infer_dataLoader = DataLoader(infer_dataset, batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)

    ########## MODEL ##############################
    model.to(device)

    ######################### Inference #########################

    all_result = []

    model.eval()
    size = len(infer_dataLoader)
    test_progress_bar = tqdm(range(size),mininterval=10)
    for text, code, labels in infer_dataLoader:
        with torch.no_grad():
            targets = labels.to(device)
            with autocast():
                outputs = model(list(text), list(code))
        all_result.extend(outputs.detach().cpu().numpy())
        test_progress_bar.update(1)

    test_data_dir = infer_file_path

    test_num = 1000
    write_result_to_file(output_infer_file, all_result, test_data_dir, test_num)





if __name__ == '__main__':
    lang = 'sql'

    inference_model_name = "../save_model/"+lang+"/model.pkl" #arg1
    infer_file_path = "../data/test/"+lang+"/batch_0.txt"  #arg2
    output_infer_file = "../results/"+lang+"/result_batch_0.txt"  #arg3 last arg


    for lr in [5e-5, 1e-5]:
        for num in [100, 500, 1000]:
            train_num = num
            # inference_model_name = "../save_model/" + lang + "/model.pkl"
            inference_model_name = "../save_model/" + lang + "/model" + str(train_num) + "_" + str(lr) + ".pkl"
            print("inference_model_name: ",inference_model_name)
            model = torch.load(inference_model_name)

            main_inference(model, infer_file_path, output_infer_file,split_num=0)

            get_mrr(lang)

            print("lr {}, train_num {}".format(lr, train_num))
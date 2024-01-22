from datasets import Dataset
from torch import nn
from transformers import AutoModel, AutoTokenizer


class Teacher_BertClassfication(nn.Module):
    def __init__(self, device):
        super(Teacher_BertClassfication, self).__init__()

        tokenizer_name = 'microsoft/codebert-base'
        # maml_model_path = "../MAML_model/checkpoint-epoch4"
        self.model = AutoModel.from_pretrained(tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.fc1 = nn.Linear(768, 2)
        self.device = device

    def forward(self, text, code):
        batch_tokenized = self.tokenizer(list(text), list(code), add_special_tokens=True,
                                    padding=True, max_length=128,
                                    truncation=True, return_tensors="pt")  # tokenize、add special token、pad

        input_ids = batch_tokenized['input_ids'].to(self.device)

        attention_mask = batch_tokenized['attention_mask'].to(self.device)

        hidden_states = self.model(input_ids, attention_mask=attention_mask, return_dict=True,
                                   output_hidden_states=True).hidden_states

        output_hidden_state = hidden_states[-1]
        outputs = output_hidden_state[:, 0, :]
        output = self.fc1(outputs)

        return output



class Teacher_Dataset_For_Lable(Dataset):
    def __init__(self, satements_code, codes, queries, labels, need_label):
        self.queries = queries
        self.codes = codes
        self.satements_codes = satements_code
        self.ls = labels
        self.need_label = need_label

        self.text_lines = []
        self.code_lines = []
        self.code_statements = []
        self.labels = []

        for q, c, l, stas in zip(self.queries, self.codes, self.ls, self.satements_codes):
            if(self.need_label == int(l)):
                self.text_lines.append(q)
                self.code_lines.append(c)
                self.code_statements.append(stas)
                self.labels.append(l)

        print("读取出 {} 的数量: {}".format(need_label, len(self.text_lines)))
    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.code_statements[i]
        d = int(self.labels[i])
        return a, b, c, d

def remove_sta(satements_codes, codes):
    all_remove_sta_codes = []
    for statement, code in zip(satements_codes, codes):
        code = code.replace('"', "").replace("'", "")
        ith_remove_sta_code = []
        for st in statement:
            st = st.replace('"', "").replace("'","")
            remove_sta_code = code.replace(st,"")

            if(remove_sta_code == code):
                print("error st: {}, ------- code: {}".format(st, code))

            # print(remove_sta_code)
            ith_remove_sta_code.append(remove_sta_code)

        all_remove_sta_codes.append(ith_remove_sta_code)

    return codes, all_remove_sta_codes

def remove_a_sta(satements_codes, code):

    code = code.replace('"', "").replace("'", "")
    ith_remove_sta_code = []
    for st in satements_codes:
        st = st.replace('"', "").replace("'","")
        remove_sta_code = code.replace(st,"")

        if(remove_sta_code == code):
            print("error st: {}, ------- code: {}".format(st, code))

        # print(remove_sta_code)
        ith_remove_sta_code.append(remove_sta_code)


    return code, ith_remove_sta_code


class Removed_Dataset_For_Lable(Dataset):
    def __init__(self, query, ith_remove_sta_codes, label):
        self.query = query
        self.codes = ith_remove_sta_codes
        self.label = label
        print("ith_remove_sta_codes的数量: {}".format(len(self.codes)))

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        a = self.query
        b = self.codes[i]
        c = self.label
        return a, b, int(c)
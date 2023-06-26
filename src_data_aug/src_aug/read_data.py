

#将从文件中读取的str形式的list，转换为真正程序用的list
#专用于codeLines，因为code中可能有单引号，需要把分割符号特殊处理
import random


def str_2_list_for_codeLines(str):
    """
    将从文件中读取的str形式的list，转换为真正程序用的list

    :param str: 形如：["SELECT name ", "FROM channel ", "WHERE OWNER != "CCTV""]
    :return:
    """
    #处理两边
    str = str[2:]
    str = str[:-2] #SELECT name ", "FROM channel ", "WHERE OWNER != "CCTV"

    #分割
    str_list = str.split('", "') # list类型的 [SELECT name, FROM channel, WHERE OWNER != "CCTV"]
    # print(str_list)

    #去除两边的''
    new_str_list = []
    for sr in str_list:
        sr = sr.replace('"',"'").strip()
        new_str_list.append(sr)

    # print(new_str_list)
    return new_str_list

def str_2_list(str):
    """
    将从文件中读取的str形式的list，转换为真正程序用的list

    :param str: 形如：['select_clause', 'from_clause', 'where_clause']
    :return:
    """
    #处理两边
    str = str[2:]
    str = str[:-2] #'select_clause', 'from_clause', 'where_clause'

    #分割
    str_list = str.split("', '") # list类型的 ['select_clause', 'from_clause', 'where_clause']

    return str_list

def read_target_data(file_path, splitnum=0):
    all_satements_code, all_parse_codes = [], []

    with open(file_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    #截断
    if (splitnum != 0):
        lines = lines[:splitnum]
    # splitnum = min(len(lines), splitnum)
    # lines = random.sample(lines, splitnum) #无放回采样

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        # if(len(temp_line)) == 5:
        all_satements_code.append(str_2_list_for_codeLines(temp_line[0]))
        all_parse_codes.append(temp_line[1])

    print("read_target_data 读取的行数：", len(all_parse_codes))
    return all_satements_code, all_parse_codes

#相对于 read_target_data，多了query
#sql版本，不用读函数申明
def read_source_data(file_path, splitnum=0):
    print("file_path: ", file_path)
    all_satements_code, all_parse_codes, all_queries, all_labels = [], [], [], []

    with open(file_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        if(int(temp_line[3]) == 1):
            # if(len(temp_line)) == 5:
            all_satements_code.append(str_2_list_for_codeLines(temp_line[0]))
            all_parse_codes.append(temp_line[1])
            all_queries.append(temp_line[2])
            all_labels.append(temp_line[3])

    # 截断
    if (splitnum != 0):
        splitnum = int(splitnum / 2)  # 除以2的原因是，对比的实验是正例+反例只和，等于few shot的训练数量
        all_satements_code = all_satements_code[:splitnum]
        all_parse_codes = all_parse_codes[:splitnum]
        all_queries = all_queries[:splitnum]
        all_labels = all_labels[:splitnum]


    print("read_source_data 读取的行数：", len(all_parse_codes))
    return all_satements_code, all_parse_codes, all_queries, all_labels


#相对于 read_target_data，多了query
#solidity版本，需要读函数申明
def read_source_data_for_solidity(file_path, splitnum=0):
    all_coarse_templates, all_middle_templates, all_fine_templates, all_satements_code, all_parse_codes\
        , all_queries, all_fuc_definitions, all_labels = [], [], [], [], [], [], [], []

    with open(file_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    #截断
    if (splitnum != 0):
        lines = lines[:splitnum]

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        # if(len(temp_line)) == 5:
        all_coarse_templates.append(str_2_list(temp_line[0]))
        all_middle_templates.append(str_2_list(temp_line[1]))
        all_fine_templates.append(str_2_list(temp_line[2]))
        all_satements_code.append(str_2_list_for_codeLines(temp_line[3]))
        all_parse_codes.append(temp_line[4])
        all_queries.append(temp_line[5])
        all_fuc_definitions.append(temp_line[6])
        all_labels.append(temp_line[7])


    print("读取的行数：", len(all_parse_codes))
    return all_coarse_templates, all_middle_templates, all_fine_templates, all_satements_code, all_parse_codes, all_queries, all_fuc_definitions, all_labels



def read_source_data_with_key_statements(file_path, splitnum=0):
    with open(file_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    if (splitnum != 0):
        lines = lines[:splitnum]

    all_labels, all_queries, all_codes, all_key_stas, all_trival_stas, all_statements = [], [], [], [], [], []
    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        all_labels.append(temp_line[0])
        all_queries.append(temp_line[3])
        all_codes.append(temp_line[4])
        all_key_stas.append(temp_line[5])
        all_trival_stas.append(temp_line[6])
        all_statements.append(str_2_list_for_codeLines(temp_line[2]))

    return all_labels, all_queries, all_codes, all_key_stas, all_trival_stas, all_statements

import execjs
import pprint

import pandas as pd
from tqdm.auto import tqdm

ctx = execjs.compile("""
const parser = require('@solidity-parser/parser');
function getAst(code){
    const input = code
    var ast = parser.parse(input, { loc: true })
        return ast
}
""") # 获取代码编译完成后的对象

def read_source_data(file_path):
    labels = []
    text_lines = []
    code_lines = []

    with open(file_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    #截断
    # lines = lines[:100]

    # 去重
    # print("1, Before Duplicates Lines: ", len(lines))
    # lines = list(set(lines))  # 需要对文件去重，否则一个模板可能对应的多个内容是一样的，影响效果，这步很重要
    # print("2, After Duplicates Lines: ", len(lines))

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        if(len(temp_line)) == 5:
            labels.append(temp_line[0])
            text_lines.append(temp_line[-2])
            code_lines.append(temp_line[-1])

    print("3, 注释和代码行数：", len(text_lines), len(code_lines) )
    return labels, text_lines, code_lines

def read_target_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # 截断
    lines = lines[:100]

    #去重
    print("1, Before Duplicates Lines: ", len(lines))
    lines = list(set(lines)) #需要对文件去重，否则一个模板可能对应的多个内容是一样的，影响效果，这步很重要
    print("2, After Duplicates Lines: ", len(lines))
    return lines

def read_excel_data(file_path):
    df = pd.read_excel(file_path, sheet_name='mybook')
    lines = df['contract'].tolist()

    #TODO: 最好把code清洗一下，contract去除掉，只留下方法体去重

    # 去重
    print("1, Before Duplicates Lines: ", len(lines))
    lines = list(set(lines))  # 需要对文件去重，否则一个模板可能对应的多个内容是一样的，影响效果，这步很重要
    print("2, After Duplicates Lines: ", len(lines))
    return lines

def write_source_data_to_file(out_file_path,
                              all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels):
    with open(out_file_path, "w") as writer:
        for code_sta, code_org, query, label in zip(all_satements_code,
                                                                   all_parse_codes, all_parse_quries, all_parse_labels):
            code_sta = str(code_sta).replace("'", '"')  # 不这样处理，会使得后续分割为list出错，但是这样处理，会影响代码的"'"
            writer.write('<CODESPLIT>'.join([str(code_sta), code_org, query, label]) + '\n')

def get_statements(sol, statements, fuc_definition):
    all_return_satements = []

    #先把函数的定义装进语句中
    all_return_satements.append(fuc_definition)

    for sta in statements:

        start = sta['loc']['start']['column']
        end = sta['loc']['end']['column']
        all_return_satements.append(sol[start:end + 1])

    return all_return_satements

def get_ast_statements_unit(sol):
    try:
        sourceUnit = ctx.call("getAst", sol)
        # pprint.pprint(sourceUnit)

    except Exception as e:
        # print("sol解析出错：", sol)
        return "parse err", ""

    # 获取每条语句，为了后续判断类型，进行处理等操作
    try:
        statements = sourceUnit['children'][0]['subNodes'][0]['body']['statements']

        #函数定义部分单独拿出来
        fuc_start = sourceUnit['children'][0]['subNodes'][0]['loc']['start']['column']
        fuc_end = sourceUnit['children'][0]['subNodes'][0]['body']['loc']['start']['column']
        fuc_definition = sol[fuc_start:fuc_end]

    except Exception as e:
        return "index err", ""
    return statements, fuc_definition

#测试函数功能
def test_fuc():
    # sol = "contract c40214{ function setSource(address a) {a=b+x; c=e-f;} }"
    # sol = "contract c40214{ function setSource(address a) { if(msg.sender!=owner)throw; a=b+x; c=e+f; logs.push(log(owner,\"setSource\",a,0)); } }"
    sol = "contract c24849{ function buyGrimReapersAgainstEther() payable returns (uint amount) { if (buyPriceEth == 0 || msg.value < buyPriceEth) throw; amount = msg.value / buyPriceEth; if (balances[this] < amount) throw; balances[msg.sender] = safeAdd(balances[msg.sender], amount); balances[this] = safeSub(balances[this], amount); Transfer(this, msg.sender, amount); return amount; } }"

    statements, fuc_definition = get_ast_statements_unit(sol)  # 对code解析出statement units
    err_parse_num = 0

    if (statements != "parse err"):

        print(get_statements(sol, statements, fuc_definition))

    else:
        err_parse_num += 1

    print("err_parse_num: ", err_parse_num)

def run():
    in_file_path = "../data_in/solidity_train.txt"
    out_file_path = "../data_out/solidity_source_statements_codes.txt"
    all_labels, all_quries, all_codes = read_source_data(in_file_path)

    err_parse_num = 0
    progress_bar = tqdm(range(len(all_codes)))
    all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels = [], [], [], []

    for label, query, code in zip(all_labels, all_quries, all_codes):
        fuc_block_statements, fuc_definition = get_ast_statements_unit(code)

        if (fuc_block_statements != "parse err" and fuc_block_statements != "index err"):
            coarse_satements = get_statements(
                code, fuc_block_statements, fuc_definition)

            all_satements_code.append(coarse_satements)
            all_parse_codes.append(code)  # 这三个satements的内容都是一样的，用一个即可
            all_parse_quries.append(query)
            # all_fuc_definitions.append(fuc_definition)
            all_parse_labels.append(label)

        else:
            err_parse_num += 1
            print("current err parse num: {}, current err type: {} ".format(err_parse_num, fuc_block_statements))
            print("parse err code: ", code)

        progress_bar.update(1)

    print("err_parse_num: ", err_parse_num)

    # 写入模板
    write_source_data_to_file(out_file_path,
                              all_satements_code, all_parse_codes, all_parse_quries, all_parse_labels)

if __name__ == '__main__':
    run()









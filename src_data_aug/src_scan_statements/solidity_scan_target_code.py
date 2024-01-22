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
""")

def read_txt_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    lines = lines[:100]

    print("1, Before Duplicates Lines: ", len(lines))
    lines = list(set(lines))
    print("2, After Duplicates Lines: ", len(lines))
    return lines

def read_excel_data(file_path):
    df = pd.read_excel(file_path, sheet_name='mybook')
    lines = df['contract'].tolist()

    print("1, Before Duplicates Lines: ", len(lines))
    lines = list(set(lines))
    print("2, After Duplicates Lines: ", len(lines))
    return lines

def write_data_to_file(out_file_path, all_satements_code, all_parse_codes):
    with open(out_file_path, "w") as writer:
        for code_sta, code_org in zip(all_satements_code, all_parse_codes):
            code_sta = str(code_sta).replace("'", '"')
            writer.write('<CODESPLIT>'.join([str(code_sta), code_org]) + '\n')


def get_statements(sol, statements, fuc_definition):
    all_return_satements = []

    all_return_satements.append(fuc_definition)

    for sta in statements:

        start = sta['loc']['start']['column']
        end = sta['loc']['end']['column']
        all_return_satements.append(sol[start:end + 1])

    return all_return_satements

def get_ast_statements_unit(sol):
    try:
        sourceUnit = ctx.call("getAst", sol)

    except Exception as e:
        return "parse err", ""

    try:
        statements = sourceUnit['children'][0]['subNodes'][0]['body']['statements']

        fuc_start = sourceUnit['children'][0]['subNodes'][0]['loc']['start']['column']
        fuc_end = sourceUnit['children'][0]['subNodes'][0]['body']['loc']['start']['column']
        fuc_definition = sol[fuc_start:fuc_end]

    except Exception as e:
        return "index err", ""
    return statements, fuc_definition

def test_fuc():
    sol = "contract c40214{ function setSource(address a) {a=b+x; c=e-f;} }"

    statements, fuc_definition = get_ast_statements_unit(sol)  # 对code解析出statement units
    err_parse_num = 0

    if (statements != "parse err"):

        print(get_statements(sol, statements, fuc_definition))
        print(fuc_definition)

    else:
        err_parse_num += 1

    print("err_parse_num: ", err_parse_num)

if __name__ == '__main__':
    test_fuc()

    in_file_path = "../data_in/solidity_unlabel_data.txt"
    out_file_path = "../data_out/solidity_target_statements_codes.txt"
    all_codes = read_txt_data(in_file_path)

    err_parse_num = 0
    progress_bar = tqdm(range(len(all_codes)))
    all_satements_code, all_parse_codes = [], []
    for code in all_codes:
        fuc_block_statements, fuc_definition = get_ast_statements_unit(code)

        if (fuc_block_statements != "parse err" and fuc_block_statements != "index err"):
            coarse_satements = get_statements(
                code, fuc_block_statements, fuc_definition)

            all_satements_code.append(coarse_satements)
            all_parse_codes.append(code)

        else:
            err_parse_num += 1
            print("current err parse num: {}, current err type: {} ".format(err_parse_num, fuc_block_statements))
            print("parse err code: ", code)

        progress_bar.update(1)

    print("err_parse_num: ", err_parse_num)

    write_data_to_file(out_file_path, all_satements_code, all_parse_codes)







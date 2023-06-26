import execjs
import pprint

import pandas as pd
from tqdm.auto import tqdm

ctx = execjs.compile("""
const parse = require('sql-parser-cst');
function getAst(query){
    const cst = parse.parse(query, {dialect: "sqlite", includeRange: true,});
    return cst
}
""")

def read_data(file_path):
    lines = []
    text_lines = []
    code_lines = []

    with open(file_path, encoding="utf=8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    #截断
    # lines = lines[:100]

    # 去重
    print("1, Before Duplicates Lines: ", len(lines))
    lines = list(set(lines))  # 需要对文件去重，否则一个模板可能对应的多个内容是一样的，影响效果，这步很重要
    print("2, After Duplicates Lines: ", len(lines))

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        text_lines.append(temp_line[1])
        code_lines.append(temp_line[0])

    print("3, 注释和代码行数：", len(text_lines), len(code_lines) )
    return text_lines, code_lines

def write_data_to_file(out_file_path, all_satements_code, all_parse_codes):
    with open(out_file_path, "w") as writer:
        for code_sta, code_org in zip(all_satements_code, all_parse_codes):
            code_sta = str(code_sta).replace("'", '"')  # 不这样处理，会使得后续分割为list出错，但是这样处理，会影响代码的"'"
            writer.write('<CODESPLIT>'.join([str(code_sta), code_org]) + '\n')



def get_ast_statements_unit(query):
    try:
        statement = ctx.call("getAst", query)
    except Exception as e:
        return "parse err"

    # 获取每条语句，为了后续判断类型，进行处理等操作
    # 分为联合语句 和 非联合语句两种
    try:
        #非联合语句的处理
        if("clauses" in statement['statements'][0].keys()):
            fuc_block_statements = statement['statements'][0]['clauses'] # 非compound_select_stmt类型的语句   可以拿到查询、更新等类型的语句
        # 联合语句的处理
        if("operator" in statement['statements'][0].keys()):
            #联合类型的SQL语句，AST的结构是：['type', 'operator', 'left', 'right', 'range']
            #由于是字典类型，不能直接转换为list进行解析，所以取 operator、left的clauses、right的clauses，拼接后送出去处理

            #left
            fuc_block_statements = statement['statements'][0]['left']['clauses']
            # operator
            operator = statement['statements'][0]['operator']  # operator是表示联合操作的类型，再追加上进行处理
            fuc_block_statements.append(operator)
            # right
            fuc_block_statements.extend(statement['statements'][0]['right']['clauses'])
    except Exception as e:
        return "index err"

    return fuc_block_statements


def get_statements(code, statements):
    all_return_satements = []

    for sta in statements:
        #对于每个语句，进行粒度解析
        #先拿到code
        start = sta['range'][0]
        end = sta['range'][1]
        all_return_satements.append(code[start:end + 1])

    return all_return_satements

#测试函数功能
def test_fuc():
    code = "SELECT foo, bar as baz FROM mytable WHERE foo LIKE '%neat%' ORDER BY foo DESC"

    fuc_block_statements = get_ast_statements_unit(code)
    print(get_statements(code, fuc_block_statements))

if __name__ == '__main__':
    test_fuc()

    in_file_path = "../data_in/sql_unlabel_data.txt"
    out_file_path = "../data_out/sql_target_templates_codes.txt"
    _, all_codes = read_data(in_file_path)

    err_parse_num = 0
    progress_bar = tqdm(range(len(all_codes)))
    all_satements_code, all_parse_codes = [], []
    for code in all_codes:
        fuc_block_statements = get_ast_statements_unit(code)

        if (fuc_block_statements != "parse err" and fuc_block_statements != "index err"):
            coarse_satements = get_statements(
                code, fuc_block_statements)

            all_satements_code.append(coarse_satements)
            all_parse_codes.append(code)  # 这三个satements的内容都是一样的，用一个即可

        else:
            err_parse_num += 1
            print("current err parse num: {}, current err type: {} ".format(err_parse_num, fuc_block_statements))
            print("parse err code: ", code)

        progress_bar.update(1)

    print("err_parse_num: ", err_parse_num)

    #写入模板
    write_data_to_file(out_file_path, all_satements_code, all_parse_codes)





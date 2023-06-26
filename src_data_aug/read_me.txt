File Structure：
./data_in: original train files and unlabeled codes files.
./data_out: augmented files
    XXX_source_statements_codes.zip: the result of using an AST lib to split original codes to statements
    XXX_target_statements_codes.zip: the result of using an AST lib to split unlabeled codes to statements
./main_src:
    main code, run main.py to augment codes
./save_model:
    teacher_url.txt: download the teacher model, which is used to recongnize key statements from codes.


./src_aug ./src_saliency
    main.py calls their functions
./src_scan_statements
    using an AST lib to split original codes to statements

RUN:
    1. unzip files in ./data_in and ./data_out
    2. run /main_src/main.py to augment codes,


NOTICE:
    augmented files are saved in /data_out/{lang}/{few-shot num},

File Structure：
./data: train, valid, and test file path
    NOTICE:
    1. augmented files need to be copied to /data/train_valid/{lang}/{few-shot num}
    2. run format_test.py to generate test files.

./fine_turn_GraBert
   file_url.txt: download a fine-tuned model, which is fine-tuned by python code search corpus.

./results: inference output files

./save_model: save trained models

./src source codes


RUN:
    1. pasting augmented files to /data/train_valid/{lang}/{few-shot num}
    2. run /src/main.py


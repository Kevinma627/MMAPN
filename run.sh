#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

python3 test_url.py --data_data_dir Test_Data/test_all.txt --data_delimit_mode 1 --data_word_dict_dir Model/runs_url/words_dict.p --data_subword_dict_dir Model/runs_url/subwords_dict.p --data_char_dict_dir Model/runs_url/chars_dict.p \
--log_checkpoint_dir Model/runs_url/checkpoints --log_output_dir Model/runs_url \
--model_emb_mode 3 --model_emb_dim 32 \
--test_batch_size 128

python3 test_html.py --data_data_dir Test_Data/test_html_all.txt --data_word_dict_dir Model/runs_html/words_dict.p --data_subword_dict_dir Model/runs_html/subwords_dict.p --data_char_dict_dir Model/runs_html/chars_dict.p \
--log_checkpoint_dir Model/runs_html/checkpoints --log_output_dir Model/runs_html \
--model_emb_dim 100 \
--test_batch_size 20

conda activate torch

python3 test_tab.py --test_data_file Test_Data/test_tab_all.csv

conda deactivate

python3 stack.py
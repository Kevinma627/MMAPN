#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

python3 test_url.py --data_data_dir Test_Data/test.txt \
--log_checkpoint_dir Model/runs_url --log_output_dir Model/runs_url \
--test_batch_size 128

python3 test_html.py --data_data_dir Test_Data/test_html_all.txt --data_word_dict_dir Model/runs_html/words_dict.p --data_subword_dict_dir Model/runs_html/subwords_dict.p --data_char_dict_dir Model/runs_html/chars_dict.p \
--log_checkpoint_dir Model/runs_html/checkpoints --log_output_dir Model/runs_html \
--model_emb_dim 100 \
--test_batch_size 20

conda activate torch # activate torch environment

python3 test_tab.py --test_data_file Test_Data/test_tab_all.csv

conda deactivate # deactivate torch environment

python3 stack.py --url_save_dir Model/runs_url --html_save_dir Model/runs_html --tab_save_dir Model/runs_tab --results_dir Final_results
#! /bin/bash
devices=$1
config=$2
# CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of cthulhu" --workspace trial_cthulhu
# CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a squirrel" --workspace trial_squirrel
# CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a cat lying on its side batting at a ball of yarn" --workspace trial_cat_lying
CUDA_VISIBLE_DEVICES=${devices} python main.py --config ${config}

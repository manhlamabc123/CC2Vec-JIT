#! /bin/bash

python jit_cc2ftr.py -train \
    -project "$1_$4" \
    -train_data "$2/$1/commits/cc2vec_$1_$4.pkl" \
    -dictionary_data "$2/$1/commits/$1_$4_dict.pkl" \
    -num_epochs $3
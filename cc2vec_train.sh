#! /bin/bash

echo "Project: $1"

python jit_cc2ftr.py -train \
    -project $1 \
    -train_data "$2/$1/cc2vec/$1_train.pkl" \
    -dictionary_data "$2/$1/cc2vec/$1_dict.pkl" \
    -num_epochs $3

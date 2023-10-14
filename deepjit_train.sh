#! /bin/bash

echo "Project: $1"


python jit_DExtended.py -train \
    -project $1 \
    -train_data "$2/$1/cc2vec/$1_train_dextend_raw.pkl" \
    -train_data_cc2ftr "extracted_features_$1.pkl" \
    -dictionary_data "$2/$1/cc2vec/$1_dict.pkl" \
    -num_epochs $3

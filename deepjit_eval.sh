#! /bin/bash

echo "Project: $1"

python jit_cc2ftr.py -predict \
    -project $1 \
    -predict_data "$2/$1/cc2vec/$1_test.pkl" \
    -dictionary_data "$2/$1/cc2vec/$1_dict.pkl" \
    -load_model "snapshot/$1/epoch_50.pt" \
    -name "extracted_features_$1.pkl"

python jit_DExtended.py -predict \
    -project $1 \
    -pred_data "$2/$1/cc2vec/$1_test_dextend_raw.pkl" \
    -pred_data_cc2ftr "extracted_features_$1.pkl" \
    -dictionary_data "$2/$1/cc2vec/$1_dict.pkl" \
    -load_model "snapshot_dextend/$1/epoch_50.pt"

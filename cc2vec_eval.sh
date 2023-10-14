#! /bin/bash

echo "Project: $1"

python jit_cc2ftr.py -predict \
    -project $1 \
    -predict_data "$2/$1/$1_train.txt" \
    -dictionary_data "$2/$1/$1_dict.pkl" \
    -load_model "snapshot/$1/epoch_50.pt" \
    -name "extracted_features_$1.pkl"

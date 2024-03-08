#! /bin/bash

python jit_cc2ftr.py -predict \
    -project "$1_$4" \
    -predict_data "$2/$1/commits/cc2vec_$1_$5.pkl" \
    -dictionary_data "$2/$1/commits/$1_$4_train_dict.pkl" \
    -load_model "snapshot/$1_$4/epoch_$3.pt" \
    -name "extracted_features_$1_$5.pkl" \
    -auc "outputs/auc.csv" \
    -testing_time "outputs/testing_time.csv" \
    -training_time "outputs/training_time.csv" \
    -ram "outputs/ram.csv" \
    -vram "outputs/vram.csv" \

python jit_DExtended.py -predict \
    -project "$1_$4" \
    -pred_data "$2/$1/commits/deepjit_$1_$5.pkl" \
    -pred_data_cc2ftr "extracted_features_$1_$5.pkl" \
    -dictionary_data "$2/$1/commits/$1_$4_train_dict.pkl" \
    -load_model "snapshot_dextend/$1_$4/epoch_$3.pt" \
    -auc "outputs/auc.csv" \
    -testing_time "outputs/testing_time.csv" \
    -training_time "outputs/training_time.csv" \
    -ram "outputs/ram.csv" \
    -vram "outputs/vram.csv" \

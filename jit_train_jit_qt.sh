#! /bin/bash

python jit_DExtended.py -train \
    -train_data "data/qt_train.pkl" \
    -train_data_cc2ftr "data/qt_train_cc2ftr.pkl" \
    -dictionary_data "data/qt_dict.pkl"
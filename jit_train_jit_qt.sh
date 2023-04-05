#! /bin/bash

python jit_DExtended.py -train \
    -train_data "../data+model/data/jit/qt_train.pkl" \
    -train_data_cc2ftr "../data+model/data/jit/qt_train_cc2ftr.pkl" \
    -dictionary_data "../data+model/data/jit/qt_dict.pkl"
#! /bin/bash

python jit_cc2ftr.py -train \
    -train_data "../data+model/data/jit/qt_train.pkl" \
    -test_data "../data+model/data/jit/qt_test.pkl" \
    -dictionary_data "../data+model/data/jit/qt_dict.pkl"
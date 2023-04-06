#! /bin/bash

python jit_cc2ftr.py -train \
    -train_data "data/qt_train.pkl" \
    -test_data "data/qt_test.pkl" \
    -dictionary_data "data/qt_dict.pkl"
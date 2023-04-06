#! /bin/bash

python jit_cc2ftr.py -predict \
    -predict_data "data/qt_test.pkl" \
    -dictionary_data "data/qt_dict.pkl" \
    -load_model "model/qt_cc2ftr.pt" \
    -name "jit_extract_qt"
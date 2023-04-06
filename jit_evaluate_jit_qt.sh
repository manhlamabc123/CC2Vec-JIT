#! /bin/bash

python jit_DExtended.py -predict \
    -pred_data "data/qt_test_dextend.pkl" \
    -pred_data_cc2ftr "data/qt_test_cc2ftr.pkl" \
    -dictionary_data "data/qt_dict.pkl" \
    -load_model "model/qt_djit_extend.pt"
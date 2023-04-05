#! /bin/bash

python jit_DExtended.py -predict \
    -pred_data "../data+model/data/jit/qt_test_dextend.pkl" \
    -pred_data_cc2ftr "../data+model/data/jit/qt_test_cc2ftr.pkl" \
    -dictionary_data "../data+model/data/jit/qt_dict.pkl" \
    -load_model "../data+model/model/jit/qt_djit_extend.pt"
#! /bin/bash

python jit_cc2ftr.py -predict \
    -predict_data "../data+model/data/jit/qt_test.pkl" \
    -dictionary_data "../data+model/data/jit/qt_dict.pkl" \
    -load_model "../data+model/model/jit/qt_cc2ftr.pt" \
    -name "jit_extract_qt"
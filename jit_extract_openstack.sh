#! /bin/bash

python jit_cc2ftr.py -predict \
    -predict_data "data/openstack_test.pkl" \
    -dictionary_data "data/openstack_dict.pkl" \
    -load_model "snapshot/2023-04-19_18-30-20/epoch_50.pt" \
    -name "jit_extract_openstack.pkl"
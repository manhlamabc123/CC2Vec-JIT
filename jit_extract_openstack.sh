#! /bin/bash

python jit_cc2ftr.py -predict \
    -predict_data "data/openstack_test.pkl" \
    -dictionary_data "data/openstack_dict.pkl" \
    -load_model "model/openstack_cc2ftr.pt" \
    -name "jit_extract_openstack"
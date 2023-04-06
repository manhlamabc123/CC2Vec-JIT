#! /bin/bash

python jit_DExtended.py -predict \
    -pred_data "data/openstack_test_dextend.pkl" \
    -pred_data_cc2ftr "data/openstack_test_cc2ftr.pkl" \
    -dictionary_data "data/openstack_dict.pkl" \
    -load_model "model/openstack_djit_extend.pt"
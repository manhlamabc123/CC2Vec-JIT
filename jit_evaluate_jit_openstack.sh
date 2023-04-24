#! /bin/bash

python jit_DExtended.py -predict \
    -pred_data "data/openstack_test_dextend.pkl" \
    -pred_data_cc2ftr "jit_extract_openstack.pkl" \
    -dictionary_data "data/openstack_dict.pkl" \
    -load_model "snapshot/2023-04-19_10-42-21/epoch_50.pt"
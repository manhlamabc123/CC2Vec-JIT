#! /bin/bash

python jit_DExtended.py -train \
    -train_data "data/openstack_train_dextend.pkl" \
    -train_data_cc2ftr "data/openstack_train_cc2ftr.pkl" \
    -dictionary_data "data/openstack_dict.pkl"
#! /bin/bash

python jit_cc2ftr.py -train \
    -train_data "data/openstack_train.pkl" \
    -test_data "data/openstack_test.pkl" \
    -dictionary_data "data/openstack_dict.pkl"
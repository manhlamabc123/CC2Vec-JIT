#! /bin/bash

python jit_cc2ftr.py -train \
    -train_data "../data+model/data/jit/openstack_train.pkl" \
    -test_data "../data+model/data/jit/openstack_test.pkl" \
    -dictionary_data "../data+model/data/jit/openstack_dict.pkl"
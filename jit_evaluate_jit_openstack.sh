#! /bin/bash

python jit_DExtended.py -predict \
    -pred_data "../data+model/data/jit/openstack_test_dextend.pkl" \
    -pred_data_cc2ftr "../data+model/data/jit/openstack_test_cc2ftr.pkl" \
    -dictionary_data "../data+model/data/jit/openstack_dict.pkl" \
    -load_model "../data+model/model/jit/openstack_djit_extend.pt"
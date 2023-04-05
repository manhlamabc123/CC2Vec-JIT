#! /bin/bash

python jit_cc2ftr.py -predict \
    -predict_data "../data+model/data/jit/openstack_test.pkl" \
    -dictionary_data "../data+model/data/jit/openstack_dict.pkl" \
    -load_model "../data+model/model/jit/openstack_cc2ftr.pt" \
    -name "jit_extract_openstack"
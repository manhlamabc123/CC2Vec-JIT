#! /bin/bash

bash cc2vec_train.sh $1

bash cc2vec_eval.sh $1

bash deepjit_train.sh $1

bash deepjit_eval.sh $1
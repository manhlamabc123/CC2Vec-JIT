#! /bin/bash

DATA_DIR="/home/manh/Documents/lapredict-paper"

bash cc2vec_train.sh $1 $DATA_DIR

bash cc2vec_eval.sh $1 $DATA_DIR

bash deepjit_train.sh $1 $DATA_DIR

bash deepjit_eval.sh $1 $DATA_DIR

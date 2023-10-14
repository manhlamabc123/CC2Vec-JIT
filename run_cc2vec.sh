#! /bin/bash

DATA_DIR="/data/gpfs/projects/punim1928/RISE/Manh/data/lapredict-paper"
EPOCHS=3

bash cc2vec_train.sh $1 $DATA_DIR $EPOCHS

bash cc2vec_eval.sh $1 $DATA_DIR $EPOCHS

bash deepjit_train.sh $1 $DATA_DIR $EPOCHS

bash deepjit_eval.sh $1 $DATA_DIR $EPOCHS

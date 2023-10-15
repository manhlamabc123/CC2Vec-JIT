#! /bin/bash

# DATA_DIR="/home/manh/Documents/Data/splited-tan-dataset"
EPOCHS=3
echo "Project: $1"
echo "Train: $2"
echo "Test: $3"

bash cc2vec_train.sh $1 $4 $EPOCHS $2

bash cc2vec_eval.sh $1 $4 $EPOCHS $2

bash deepjit_train.sh $1 $4 $EPOCHS $2
 
bash deepjit_eval.sh $1 $4 $EPOCHS $2 $3

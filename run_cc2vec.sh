#! /bin/bash

# DATA_DIR="/home/manh/Documents/Data/splited-tan-dataset"
echo "Project: $1"
echo "Train: $2"
echo "Test: $3"
echo "Epochs: $5"
echo "Data dir: $4"

bash cc2vec_train.sh $1 $4 $5 $2

bash cc2vec_eval.sh $1 $4 $5 $2

bash deepjit_train.sh $1 $4 $5 $2

echo "bash deepjit_eval.sh $1 $4 $5 $2 $3"
bash deepjit_eval.sh $1 $4 $5 $2 $3

# My-CC2Vec-JIT-

## How to run

### Branches
```
git checkout [branch]
```
* codeBERT_first: using `output[0][:, 0]`
* codeBERT_last: using `output[0][:, -1]`
* codeBERT_max: using `torch.max()`
* codeBERT: using `torch.mean()`

### To train CC2Vec
```
python jit_cc2ftr.py -train \
    -train_data "path to train data" \
    -dictionary_data "path to dictionary"
```

### Use CC2Vec to extract code features for training DeepJIT
```
python jit_cc2ftr.py -predict \
    -predict_data "path to train data" \
    -dictionary_data "path to dictionary" \
    -load_model "path pretrained CC2Vec" \
    -name "extracted_features.pkl"
```

### To train DeepJIT with new extracted features
```
python jit_DExtended.py -train \
    -train_data "path to dextend train data" \
    -train_data_cc2ftr "extracted_features.pkl" \
    -dictionary_data "data/openstack_dict.pkl"
```

### To evaluate DeepJIT with CC2Vec
```
python jit_cc2ftr.py -predict \
    -predict_data "path to test data" \
    -dictionary_data "path to dictionary" \
    -load_model "pretrained CC2Vec" \
    -name "extracted_features.pkl"

python jit_DExtended.py -predict \
    -pred_data "path to dextend test data" \
    -pred_data_cc2ftr "extracted_features.pkl" \
    -dictionary_data "path to dictionary" \
    -load_model "pretrained DeepJIT"
```

import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import T5Tokenizer
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, added_code_list, removed_code_list, tokenizer, pad_token_id, labels, max_seq_length):
        self.added_code_list = added_code_list
        self.removed_code_list = removed_code_list
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.labels = labels
    
    def __len__(self):
        return len(self.added_code_list)
    
    def __getitem__(self, idx):
        added_code = self.added_code_list[idx]

        removed_code = self.removed_code_list[idx]

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'added_code': added_code,
            'removed_code': removed_code,
            'labels': labels
        }

    def collate_fn(self, batch):
        input_data = {
            'added_code': [item['added_code'] for item in batch],
            'removed_code': [item['removed_code'] for item in batch]
        }

        # tokenize and pad the code sequences
        added_code_tokenized = self.tokenizer.batch_encode_plus(
            input_data['added_code'],
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length
        )

        # tokenize and pad the code sequences
        removed_code_tokenized = self.tokenizer.batch_encode_plus(
            input_data['removed_code'],
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length
        )
        
        # create the labels tensor
        labels = torch.stack([item['labels'] for item in batch])

        # return the tokenized input data and labels
        return {
            'added_code': {k: v.squeeze(0) for k, v in added_code_tokenized.items()},
            'removed_code': {k: v.squeeze(0) for k, v in removed_code_tokenized.items()},
            'labels': labels
        }
    
def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return ' '.join([line_split[i] for i in range(max_length)])
    else:
        return line
    
def convert_msg_to_label(pad_msg, dict_msg):
    nrows, ncols = pad_msg.shape
    labels = []
    for i in range(nrows):
        column = list(set(list(pad_msg[i, :])))
        label = np.zeros(len(dict_msg))
        for c in column:
            label[c] = 1
        labels.append(label)
    return np.array(labels)

def mapping_dict_msg(pad_msg, dict_msg):
    return np.array(
        [np.array([dict_msg[w.lower()] if w.lower() in dict_msg.keys() else dict_msg['<NULL>'] for w in line.split(' ')]) for line in pad_msg])

def padding_message(data, max_length):
    return [padding_length(line=d, max_length=max_length) for d in data]

def preprocess_data(params, max_seq_length: int = 512):
    if params.train is True:
        # Load train data
        train_data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, msgs, codes = train_data
    
    elif params.predict is True:
        # Load predict data
        predict_data = pickle.load(open(params.predict_data, 'rb'))
        ids, labels, msgs, codes = predict_data

    # Load dictionary
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary  

    # Combine train data and test data into data
    ids = ids
    labels = list(labels)
    msgs = msgs
    codes = codes

    # Handling messages
    pad_msg = padding_message(data=msgs, max_length=params.msg_length)
    pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)

    # CodeBERT tokenizer
    model = T5Model.from_pretrained("Salesforce/codet5-base")

    # Preprocessing codes
    added_list = []
    removed_list = []

    for cc_idx in tqdm(range(len(codes))): 
        added_code = ""
        removed_code = ""
        for i in range(len(train_codes[cc_idx])):
            if codes[cc_idx][i]['added_code'] != []:
                added_code += codes[cc_idx][i]['added_code'][0] + "</s>"
            if codes[cc_idx][i]['removed_code'] != []:
                removed_code += codes[cc_idx][i]['removed_code'][0] + "</s>"
        added_list.append(added_code)
        removed_list.append(removed_code)

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(added_code_list, removed_code_list, pad_msg_labels, tokenizer, max_seq_length)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size, collate_fn=code_dataset.collate_fn, shuffle=False)

    return (code_dataloader, pad_msg_labels, dict_msg, dict_code)
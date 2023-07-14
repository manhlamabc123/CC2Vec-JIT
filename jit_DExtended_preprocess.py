import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import RobertaTokenizer
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, cc2ftr, code_list, message_list, pad_token_id, labels, max_seq_length):
        self.code_list = code_list
        self.cc2ftr = cc2ftr
        self.message_list = message_list
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.labels = labels
    
    def __len__(self):
        return len(self.code_list)
    
    def __getitem__(self, idx):
        # truncate the code sequence if it exceeds max_seq_length
        code = self.code_list[idx][:self.max_seq_length]
        
        # pad the code sequence if it is shorter than max_seq_length
        num_padding = self.max_seq_length - len(code)
        code += [self.pad_token_id] * num_padding

        message = self.message_list[idx]

        cc2ftr = self.cc2ftr[idx]

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        code = torch.tensor(code)
        message = torch.tensor(message)
        cc2ftr = torch.from_numpy(cc2ftr)

        return {
            'code': code,
            'message': message,
            'feature': cc2ftr,
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
        ids, msgs, codes, labels = train_data
        data_ftr = pickle.load(open(params.train_data_cc2ftr, 'rb'))
    
    elif params.predict is True:
        # Load predict data
        predict_data = pickle.load(open(params.pred_data, 'rb'))
        ids, msgs, codes, labels = predict_data
        data_ftr = pickle.load(open(params.pred_data_cc2ftr, 'rb'))

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

    # CodeBERT tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # Tokenize codes
    code_list = []

    for commit in codes:
        code_tokens = [tokenizer.cls_token]
        for hunk in commit:
            code_tokens += tokenizer.tokenize(hunk) + [tokenizer.sep_token]
        code_tokens += [tokenizer.eos_token]
        tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        code_list.append(tokens_ids)

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(data_ftr, code_list, pad_msg, tokenizer.pad_token_id, labels, max_seq_length)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size)

    return (data_ftr, code_dataloader, dict_msg, dict_code)
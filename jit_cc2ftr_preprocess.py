import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import RobertaTokenizer
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, codes, pad_token_id, labels, max_seq_length):
        self.codes = codes
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.labels = labels
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = self.codes[idx]
        for hunk in code:
            for line in hunk["added_code"]:

        # pad the code sequence if it is shorter than max_seq_length
        num_padding = self.max_seq_length - len(added_code)
        added_code += [self.pad_token_id] * num_padding

        # truncate the code sequence if it exceeds max_seq_lengthadded
        removed_code = self.removed_code_list[idx][:self.max_seq_length]
        
        # pad the code sequence if it is shorter than max_seq_length
        num_padding = self.max_seq_length - len(removed_code)
        removed_code += [self.pad_token_id] * num_padding

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        added_code = torch.tensor(added_code)
        removed_code = torch.tensor(removed_code)

        return {
            'added_code': added_code,
            'removed_code': removed_code,
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

    labels = list(labels)

    # Load dictionary
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary  

    # Handling messages
    pad_msg = padding_message(data=msgs, max_length=params.msg_length)
    pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)

    # CodeBERT tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print(codes[1])

    for commit in codes:
        for hunk in commit:
            added_lines = []
            removed_lines = []
            for line in hunk["added_code"]:
                line_tokens = [tokenizer.cls_token] + tokenizer.tokenize(line) + [tokenizer.eos_token]
                line_tokens_ids = tokenizer.convert_tokens_to_ids(line_tokens)
                line_tokens_ids = line_tokens_ids[:max_seq_length]
                num_padding = max_seq_length - len(line_tokens_ids)
                line_tokens_ids += [tokenizer.pad_token_id] * num_padding
                added_lines.append(line_tokens_ids)
            hunk["added_code"] = added_lines
            for line in hunk["removed_code"]:
                line_tokens = [tokenizer.cls_token] + tokenizer.tokenize(line) + [tokenizer.eos_token]
                line_tokens_ids = tokenizer.convert_tokens_to_ids(line_tokens)
                line_tokens_ids = line_tokens_ids[:max_seq_length]
                num_padding = max_seq_length - len(line_tokens_ids)
                line_tokens_ids += [tokenizer.pad_token_id] * num_padding
                removed_lines.append(line_tokens_ids)
            hunk["removed_code"] = removed_lines

    print(codes[1])

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(codes, pad_msg_labels, tokenizer.pad_token_id, max_seq_length)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size)

    return (code_dataloader, pad_msg_labels, dict_msg, dict_code)
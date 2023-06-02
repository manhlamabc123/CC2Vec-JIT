import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from jit_padding import *

class CustomDataset(Dataset):
    def __init__(self, added_codes, removed_codes, labels):
        self.added_codes = added_codes
        self.removed_codes = removed_codes
        self.labels = labels
    
    def __len__(self):
        return len(self.added_codes)
    
    def __getitem__(self, idx):
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        added_code = torch.tensor(self.added_codes[idx])
        removed_code = torch.tensor(self.removed_codes[idx])

        return {
            'added_code': added_code,
            'removed_code': removed_code,
            'labels': labels
        }
    
def preprocess_data(params):
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

    # Handling codes
    added_code, removed_code = clean_and_reformat_code(codes)

    pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
    pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
    pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
    pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(pad_added_code, pad_removed_code, pad_msg_labels)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size)

    return (code_dataloader, pad_msg_labels, dict_msg, dict_code)
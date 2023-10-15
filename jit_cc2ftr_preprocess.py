import pickle, torch
from torch.utils.data import Dataset, DataLoader
from jit_padding import padding_message, clean_and_reformat_code, padding_commit_code, mapping_dict_msg, mapping_dict_code, convert_msg_to_label

class CustomDataset(Dataset):
    def __init__(self, pad_added_code, pad_removed_code, pad_msg_labels):
        self.pad_added_code = pad_added_code
        self.pad_removed_code = pad_removed_code
        self.pad_msg_labels = pad_msg_labels
    
    def __len__(self):
        return len(self.pad_added_code)
    
    def __getitem__(self, idx):
        added_code = self.pad_added_code[idx]

        removed_code = self.pad_removed_code[idx]

        msg_labels = self.pad_msg_labels[idx]

        added_code = torch.tensor(added_code)
        removed_code = torch.tensor(removed_code)
        msg_labels = torch.tensor(msg_labels)

        return {
            'added_code': added_code,
            'removed_code': removed_code,
            'msg_labels': msg_labels
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

    # Load dictionary
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary  

    # Combine train data and test data into data
    ids = ids
    labels = list(labels)
    msgs = msgs
    codes = codes

    # Preprocessing code & msg
    pad_msg = padding_message(data=msgs, max_length=params.msg_length)
    added_code, removed_code = clean_and_reformat_code(codes)
    pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
    pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)

    pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
    pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(pad_added_code, pad_removed_code, pad_msg_labels)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size, drop_last=True)

    return (code_dataloader, pad_msg_labels, dict_msg, dict_code)
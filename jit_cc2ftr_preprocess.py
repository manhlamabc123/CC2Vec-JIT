import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import RobertaTokenizer

class CustomDataset(Dataset):
    def __init__(self, added_code_list, removed_code_list, pad_token_id, labels, max_seq_length):
        self.added_code_list = added_code_list
        self.removed_code_list = removed_code_list
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.labels = labels
    
    def __len__(self):
        return len(self.added_code_list)
    
    def __getitem__(self, idx):
        # truncate the code sequence if it exceeds max_seq_length
        added_code = self.added_code_list[idx][:self.max_seq_length]
        
        # pad the code sequence if it is shorter than max_seq_length
        num_padding = self.max_seq_length - len(added_code)
        added_code += [self.pad_token_id] * num_padding

        # truncate the code sequence if it exceeds max_seq_lengthadded
        removed_code = self.removed_code_list[idx][:self.max_seq_length]
        
        # pad the code sequence if it is shorter than max_seq_length
        num_padding = self.max_seq_length - len(removed_code)
        removed_code += [self.pad_token_id] * num_padding

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        added_code = torch.tensor(added_code)
        removed_code = torch.tensor(removed_code)

        return {
            'added_code': added_code,
            'removed_code': removed_code,
            'label': label
        }

def preprocess_data(params, max_seq_length: int = 512):
    # Load train data
    train_data = pickle.load(open(params.train_data, 'rb'))
    train_ids, train_labels, train_messages, train_codes = train_data

    # Load test data
    test_data = pickle.load(open(params.test_data, 'rb'))
    test_ids, test_labels, test_messages, test_codes = test_data        

    # Load dictionary
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary  

    # Combine train data and test data into data
    ids = train_ids + test_ids
    labels = list(train_labels) + list(test_labels)
    msgs = train_messages + test_messages
    codes = train_codes + test_codes

    # CodeBERT tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # Preprocessing codes
    added_code_list = []
    removed_code_list = []

    for commit in codes:
        added_code_tokens = [tokenizer.cls_token]
        removed_code_tokens = [tokenizer.cls_token]
        for hunk in commit:
            added_code = " ".join(hunk["added_code"])
            removed_code = " ".join(hunk["removed_code"])
            added_code_tokens += tokenizer.tokenize(added_code) + [tokenizer.sep_token]
            removed_code_tokens += tokenizer.tokenize(removed_code) + [tokenizer.sep_token]
        added_code_tokens += [tokenizer.eos_token]
        removed_code_tokens += [tokenizer.eos_token]
        added_tokens_ids = tokenizer.convert_tokens_to_ids(added_code_tokens)
        removed_tokens_ids = tokenizer.convert_tokens_to_ids(removed_code_tokens)
        # added_tokens_tensor = torch.tensor(added_tokens_ids, device="cuda")
        # removed_tokens_tensor = torch.tensor(removed_tokens_ids, device="cuda")
        added_code_list.append(added_tokens_ids)
        removed_code_list.append(removed_tokens_ids)

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(added_code_list, removed_code_list, tokenizer.pad_token_id, labels, max_seq_length)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size)

    return (code_dataloader, dict_msg, dict_code)
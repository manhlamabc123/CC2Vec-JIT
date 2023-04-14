from torch.utils.data import Dataset
from global_variables import *

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

        label = torch.tensor(self.labels[idx], dtype=torch.float32, device=DEVICE)
        added_code = torch.tensor(added_code, device=DEVICE)
        removed_code = torch.tensor(removed_code, device=DEVICE)

        return {
            'added_code': added_code,
            'removed_code': removed_code,
            'label': label
        }
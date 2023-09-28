import pickle, torch
from torch.utils.data import Dataset, DataLoader
from jit_DExtended_padding import padding_data

class CustomDataset(Dataset):
    def __init__(self, cc2ftr, code_list, message_list, labels):
        self.code_list = code_list
        self.cc2ftr = cc2ftr
        self.message_list = message_list
        self.labels = labels
    
    def __len__(self):
        return len(self.code_list)
    
    def __getitem__(self, idx):
        code = self.code_list[idx]
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

def preprocess_data(params):
    if params.train is True:
        # Load train data
        train_data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, msgs, codes = train_data
        data_ftr = pickle.load(open(params.train_data_cc2ftr, 'rb'))
    
    elif params.predict is True:
        # Load predict data
        predict_data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = predict_data
        data_ftr = pickle.load(open(params.pred_data_cc2ftr, 'rb'))

    # Load dictionary
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary  

    # Combine train data and test data into data
    ids = ids
    labels = list(labels)
    msgs = msgs
    codes = codes

    pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')        
    pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')

    # Using Pytorch Dataset and DataLoader
    code_dataset = CustomDataset(data_ftr, pad_code, pad_msg, labels)
    code_dataloader = DataLoader(code_dataset, batch_size=params.batch_size, drop_last=True)

    return (data_ftr, code_dataloader, dict_msg, dict_code)
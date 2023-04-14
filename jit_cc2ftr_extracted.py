from jit_cc2ftr_model import HierachicalRNN
from jit_utils import mini_batches
import torch
from tqdm import tqdm
import pickle
from global_variables import *

def extracted_cc2ftr(data, params):
    # pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    # batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels, 
    #                         mini_batch_size=params.batch_size, shuffled=False)
    # params.vocab_code = len(dict_code)
    # params.class_num = 1 if len(labels.shape) == 1 else labels.shape[1]

    code_loader, dict_msg, dict_code = data
    # batches = mini_batches(X_added_code=None, X_removed_code=None, Y=pad_msg_labels, mini_batch_size=params.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.vocab_code = len(dict_code)    

    params.class_num = 1

    # Device configuration
    params.device = DEVICE
    model = HierachicalRNN(args=params)
    model.load_state_dict(torch.load(params.load_model))
    if torch.cuda.is_available():
        model = model.to(device=DEVICE)

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    commit_ftrs = []
    with torch.no_grad():
        for batch in tqdm(code_loader):
            added_code = batch["added_code"]
            removed_code = batch["removed_code"]
            label = batch["label"]

            commit_ftr = model.forward_commit_embeds_diff(added_code, removed_code)
            commit_ftrs.append(commit_ftr)
        commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()
    pickle.dump(commit_ftrs, open(params.name, 'wb'))
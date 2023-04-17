from jit_cc2ftr_model import HierachicalRNN
import torch
from tqdm import tqdm
import pickle

def extracted_cc2ftr(data, params):
    # Split data
    code_loader, _, dict_code = data

    # Set up param
    params.vocab_code = len(dict_code)    
    params.class_num = 1

    # Device configuration
    model = HierachicalRNN(args=params)
    model.load_state_dict(torch.load(params.load_model)).to(params.device)

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    commit_ftrs = []
    with torch.no_grad():
        for batch in tqdm(code_loader):
            added_code = batch["added_code"].to(params.device)
            removed_code = batch["removed_code"].to(params.device)

            commit_ftr = model.forward_commit_embeds_diff(added_code, removed_code)
            commit_ftrs.append(commit_ftr)

        commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()

    pickle.dump(commit_ftrs, open(params.name, 'wb'))
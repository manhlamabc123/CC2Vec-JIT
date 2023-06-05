from jit_cc2ftr_model import HierachicalRNN
import torch
from tqdm import tqdm
import pickle

def extracted_cc2ftr(data, params):
    # Split data
    code_loader, pad_msg_labels, _, dict_code = data

    # Set up param
    params.vocab_code = len(dict_code)    
    if len(pad_msg_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = pad_msg_labels.shape[1]

    # Device configuration
    model = HierachicalRNN(args=params).to(params.device)
    # model = torch.compile(model, backend="inductor")
    model.load_state_dict(torch.load(params.load_model, map_location=params.device), strict=False)

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    commit_ftrs = []
    with torch.no_grad():
        for batch in tqdm(code_loader):
            added_code = batch["added_code"].to(params.device)
            removed_code = batch["removed_code"].to(params.device)

            if model.batch_size != added_code.shape[0]:
                model.batch_size = added_code.shape[0]

            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()  

            commit_ftr = model.forward_commit_embeds_diff(added_code, removed_code, state_hunk, state_sent, state_word)
            commit_ftrs.append(commit_ftr)

        commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()

    pickle.dump(commit_ftrs, open(params.name, 'wb'))
import torch, pickle, time
from jit_cc2ftr_model import HierachicalRNN
from tqdm import tqdm
from jit_utils import *
from logs import *

def extracted_cc2ftr(data, params):
    code_loader, labels, _, dict_code = data

    params.vocab_code = len(dict_code)
    params.class_num = 1 if len(labels.shape) == 1 else labels.shape[1]

    model = HierachicalRNN(args=params).to(device=params.device)
    model.load_state_dict(torch.load(params.load_model, map_location=params.device), strict=False)

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    commit_ftrs = []
    with torch.no_grad():
        # Record the start time
        start_time = time.time()

        for batch in tqdm(code_loader):
            state_word = model.init_hidden_word(params.device)
            state_sent = model.init_hidden_sent(params.device)
            state_hunk = model.init_hidden_hunk(params.device)

            pad_added_code = batch['added_code'].to(params.device)
            pad_removed_code = batch['removed_code'].to(params.device)

            commit_ftr = model.forward_commit_embeds_diff(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)

            commit_ftrs.append(commit_ftr)
            
        commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()

        # Record the end time
        end_time = time.time()

    pickle.dump(commit_ftrs, open(params.name, 'wb'))

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    ram = get_ram_usage()

    vram = get_vram_usage()

    # Call the function to write the content to the file
    logs(params.testing_time, params.project, elapsed_time, "CC2Vec")
    logs(params.ram, params.project, ram, "CC2Vec")
    logs(params.vram, params.project, vram, "CC2Vec")
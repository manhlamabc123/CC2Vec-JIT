from jit_utils import save
import torch
import os
import torch.nn as nn
from jit_cc2ftr_model import HierachicalRNN

def train_model(data, params):
    code_loader, pad_msg_labels, _, dict_code = data

    params.save_dir = os.path.join(params.save_dir, params.project)
    params.vocab_code = len(dict_code)
    if len(pad_msg_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = pad_msg_labels.shape[1]

    # Device configuration
    model = HierachicalRNN(args=params).to(device=params.device)
    if params.load_model != None:
        model.load_state_dict(torch.load(params.load_model, map_location=params.device), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for batch in code_loader:
            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word(params.device)
            state_sent = model.init_hidden_sent(params.device)
            state_hunk = model.init_hidden_hunk(params.device)

            pad_added_code = batch['added_code'].to(params.device)
            pad_removed_code = batch['removed_code'].to(params.device)
            labels = batch['msg_labels'].to(params.device)

            optimizer.zero_grad()
            predict = model(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)
            loss = criterion(predict, labels)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        save(model, params.save_dir, 'epoch', epoch)
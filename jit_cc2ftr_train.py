from jit_utils import save
import torch
import os, datetime
import torch 
import torch.nn as nn
from tqdm import tqdm
from jit_cc2ftr_model import HierachicalRNN

def train_model(data, params):
    # Split data
    code_loader, pad_msg_labels, _, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)    
    if len(pad_msg_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = pad_msg_labels.shape[1]

    # Create model, optimizer, criterion
    model = HierachicalRNN(args=params).to(params.device)
    # model = torch.compile(model, backend="inductor")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            added_code = batch["added_code"].to(params.device)
            removed_code = batch["removed_code"].to(params.device)
            labels = batch["labels"].to(params.device)

            # Set model batch_size
            if added_code.shape[0] != model.batch_size:
                model.batch_size = added_code.shape[0]
                print(model.batch_size)

                # reset the hidden state of hierarchical attention model
                state_word = model.init_hidden_word()
                state_sent = model.init_hidden_sent()
                state_hunk = model.init_hidden_hunk()
                print(state_word.size())
                
                optimizer.zero_grad()

                # Forward
                predict = model(added_code, removed_code, state_hunk, state_sent, state_word)

                # Calculate loss
                loss = criterion(predict, labels)

                loss.backward()

                total_loss += loss

                optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))

        save(model, params.save_dir, 'epoch', epoch)
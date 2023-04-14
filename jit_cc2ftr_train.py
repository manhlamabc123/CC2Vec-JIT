from jit_utils import mini_batches, save
import torch
import os, datetime
import torch 
import torch.nn as nn
from tqdm import tqdm
from jit_cc2ftr_model import HierachicalRNN
from global_variables import *

def train_model(data, params):
    code_loader, dict_msg, dict_code = data
    # batches = mini_batches(X_added_code=None, X_removed_code=None, Y=pad_msg_labels, mini_batch_size=params.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)    

    params.class_num = 1

    # Device configuration
    params.device = DEVICE
    model = HierachicalRNN(args=params)
    if torch.cuda.is_available():
        model = model.to(params.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCEWithLogitsLoss()
    # batches = batches[:10]
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for batch in tqdm(code_loader):
            added_code = batch["added_code"]
            removed_code = batch["removed_code"]
            label = batch["label"]
            # # reset the hidden state of hierarchical attention model
            # state_word = model.init_hidden_word()
            # state_sent = model.init_hidden_sent()
            # state_hunk = model.init_hidden_hunk()

            # pad_added_code, pad_removed_code, labels = batch
            # labels = torch.cuda.FloatTensor(labels)
            # pad_added_code = torch.tensor(pad_added_code, device=params.device)
            # pad_removed_code = torch.tensor(pad_removed_code, device=params.device)
            # pad_added_code = pad_added_code.view(params.batch_size, -1)
            # pad_removed_code = pad_removed_code.view(params.batch_size, -1)
            # labels = torch.tensor(labels, dtype=torch.long, device=params.device)
            optimizer.zero_grad()
            predict = model.forward(added_code, removed_code)
            loss = criterion(predict, label)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        save(model, params.save_dir, 'epoch', epoch)
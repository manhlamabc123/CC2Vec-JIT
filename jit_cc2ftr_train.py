from jit_utils import save
import torch
import os, datetime
import torch 
import torch.nn as nn
from tqdm import tqdm
from jit_cc2ftr_model import HierachicalRNN

def train_model(data, params):
    # Split data
    code_loader, _, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)    
    params.class_num = 1

    # Create model, optimizer, criterion
    model = HierachicalRNN(args=params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            added_code = batch["added_code"].to(params.device)
            removed_code = batch["removed_code"].to(params.device)
            label = batch["label"].to(params.device)
            
            optimizer.zero_grad()

            # Forward
            predict = model.forward(added_code, removed_code)

            # Calculate loss
            loss = criterion(predict, label)

            loss.backward()

            total_loss += loss

            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))

        save(model, params.save_dir, 'epoch', epoch)
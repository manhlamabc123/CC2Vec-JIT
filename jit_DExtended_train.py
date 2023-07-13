from jit_DExtended_model import DeepJITExtended
import torch 
from tqdm import tqdm
from jit_utils import mini_batches_update_DExtended
import torch.nn as nn
import os, datetime
from jit_utils import save

def train_model(data, params):
    cc2ftr, code_loader, dict_msg, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, params.project)
    params.vocab_code = len(dict_code)
    params.vocab_msg = len(dict_msg)
    params.class_num = 1
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.feature_size = cc2ftr[0].shape[0]

    # create and train the defect model
    model = DeepJITExtended(args=params).to(device=params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()

    # Training
    for epoch in range(1, params.num_epochs + 1):
        if epoch <= 2:
            for param in model.codeBERT.parameters():
                param.requires_grad = True
        else:
            for param in model.codeBERT.parameters():
                param.requires_grad = False
                
        total_loss = 0
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            cc2ftr = batch["feature"].to(params.device)
            
            optimizer.zero_grad()

            # Forward
            predict = model(cc2ftr, code, message)

            # Calculate loss
            loss = criterion(predict, labels)

            loss.backward()

            total_loss += loss

            optimizer.step()

        print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        save(model, params.save_dir, 'epoch', epoch)

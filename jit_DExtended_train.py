from jit_DExtended_model import DeepJITExtended
import torch, os, time
from tqdm import tqdm
import torch.nn as nn
from jit_utils import *
from logs import *

def train_model(data, params):
    cc2ftr, code_loader, dict_msg, dict_code = data

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, params.project)
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.embedding_ftr = cc2ftr.shape[1]
    params.class_num = 1

    # create and train the defect model
    model = DeepJITExtended(args=params).to(device=params.device)
    if params.load_model != None:
        model.load_state_dict(torch.load(params.load_model, map_location=params.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()

    # Record the start time
    start_time = time.time()

    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            cc2ftr = batch["feature"].to(params.device)

            optimizer.zero_grad()

            # Forward
            predict = model(cc2ftr, message, code)

            # Calculate loss
            loss = criterion(predict, labels)

            loss.backward()

            total_loss += loss

            optimizer.step()

        print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        save(model, params.save_dir, 'epoch', epoch)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    ram = get_ram_usage()

    vram = get_vram_usage()

    # Call the function to write the content to the file
    logs(params.training_time, params.project, elapsed_time, "DeepJIT Extend")
    logs(params.ram, params.project, ram, "DeepJIT Extend")
    logs(params.vram, params.project, vram, "DeepJIT Extend")
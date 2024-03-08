import numpy as np
import math, os, torch, random, psutil, subprocess
import pandas as pd

def log_auc(file_path, project_name, auc):
    # Read existing CSV file
    df = pd.read_csv(file_path)
    
    # Append new data to DataFrame
    filtered_df = df[df["Project Name"] == project_name]
    
    # Update the cell corresponding to 'CC2Vec' column with the AUC score
    if not filtered_df.empty:
        df.at[filtered_df.index[0], "CC2Vec"] = auc
        # If you expect multiple rows with the same project_name, you may want to iterate over filtered_df to update all corresponding rows
    else:
        # If filtered DataFrame is empty, create a new row
        new_row = {"Project Name": project_name, "CC2Vec": auc, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0}  # Update with actual column names
        df = df._append(new_row, ignore_index=True)

    # Write DataFrame back to CSV file
    df.to_csv(file_path, index=False)

def log_testing_time(file_path, project_name, testing_time, model):
    # Read existing CSV file
    df = pd.read_csv(file_path)
    
    # Append new data to DataFrame
    filtered_df = df[df["Project Name"] == project_name]
    
    # Update the cell corresponding to 'CC2Vec' column with the AUC score
    if not filtered_df.empty:
        df.at[filtered_df.index[0], model] = testing_time
        # If you expect multiple rows with the same project_name, you may want to iterate over filtered_df to update all corresponding rows
    else:
        # If filtered DataFrame is empty, create a new row
        if model == "CC2Vec":
            new_row = {"Project Name": project_name, "CC2Vec": testing_time, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": 0.0}  # Update with actual column names
        else:
            new_row = {"Project Name": project_name, "CC2Vec": 0.0, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": testing_time}  # Update with actual column names
        df = df._append(new_row, ignore_index=True)

    # Write DataFrame back to CSV file
    df.to_csv(file_path, index=False)

def log_training_time(file_path, project_name, training_time, model):
    # Read existing CSV file
    df = pd.read_csv(file_path)
    
    # Append new data to DataFrame
    filtered_df = df[df["Project Name"] == project_name]
    
    # Update the cell corresponding to 'CC2Vec' column with the AUC score
    if not filtered_df.empty:
        df.at[filtered_df.index[0], model] = training_time
        # If you expect multiple rows with the same project_name, you may want to iterate over filtered_df to update all corresponding rows
    else:
        # If filtered DataFrame is empty, create a new row
        if model == "CC2Vec":
            new_row = {"Project Name": project_name, "CC2Vec": training_time, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": 0.0}  # Update with actual column names
        else:
            new_row = {"Project Name": project_name, "CC2Vec": 0.0, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": training_time}  # Update with actual column names
        df = df._append(new_row, ignore_index=True)

    # Write DataFrame back to CSV file
    df.to_csv(file_path, index=False)

def log_ram(file_path, project_name, ram, model):
    # Read existing CSV file
    df = pd.read_csv(file_path)
    
    # Append new data to DataFrame
    filtered_df = df[df["Project Name"] == project_name]
    
    # Update the cell corresponding to 'CC2Vec' column with the AUC score
    if not filtered_df.empty:
        df.at[filtered_df.index[0], model] = ram
        # If you expect multiple rows with the same project_name, you may want to iterate over filtered_df to update all corresponding rows
    else:
        # If filtered DataFrame is empty, create a new row
        if model == "CC2Vec":
            new_row = {"Project Name": project_name, "CC2Vec": ram, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": 0.0}  # Update with actual column names
        else:
            new_row = {"Project Name": project_name, "CC2Vec": 0.0, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": ram}  # Update with actual column names
        df = df._append(new_row, ignore_index=True)

    # Write DataFrame back to CSV file
    df.to_csv(file_path, index=False)

def log_vram(file_path, project_name, vram, model):
    # Read existing CSV file
    df = pd.read_csv(file_path)
    
    # Append new data to DataFrame
    filtered_df = df[df["Project Name"] == project_name]
    
    # Update the cell corresponding to 'CC2Vec' column with the AUC score
    if not filtered_df.empty:
        df.at[filtered_df.index[0], model] = vram
        # If you expect multiple rows with the same project_name, you may want to iterate over filtered_df to update all corresponding rows
    else:
        # If filtered DataFrame is empty, create a new row
        if model == "CC2Vec":
            new_row = {"Project Name": project_name, "CC2Vec": vram, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": 0}  # Update with actual column names
        else:
            new_row = {"Project Name": project_name, "CC2Vec": 0.0, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": 0.0, "DeepJIT Extend": vram}  # Update with actual column names
        df = df._append(new_row, ignore_index=True)

    # Write DataFrame back to CSV file
    df.to_csv(file_path, index=False)
    
def get_ram_usage():
    # Get system memory usage in bytes
    memory_usage = psutil.virtual_memory().used
    memory_usage_mb = memory_usage / (1024 * 1024)
    return memory_usage_mb

def get_vram_usage():
    # Get GPU memory usage using nvidia-smi command
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits'])
        vram_usage = int(output.strip().split(b'\n')[1])
        return vram_usage
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # Handle case where nvidia-smi is not available or GPU is not Nvidia

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = f'{save_prefix}_{epochs}.pt'
    torch.save(model.state_dict(), save_path)

def mini_batches(X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0, shuffled=True):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    if shuffled == True:
        permutation = list(np.random.permutation(m))
        shuffled_X_added = X_added_code[permutation, :, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :, :]

        shuffled_Y = Y[permutation] if len(Y.shape) == 1 else Y[permutation, :]
    else:
        shuffled_X_added = X_added_code
        shuffled_X_removed = X_removed_code
        shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(num_complete_minibatches):                
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches_DExtended(X_ftr, X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    shuffled_X_ftr, shuffled_X_msg, shuffled_X_code, shuffled_Y = X_ftr, X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(num_complete_minibatches):
        mini_batch_X_ftr = shuffled_X_ftr[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_ftr = shuffled_X_ftr[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches_update_DExtended(X_ftr, X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_ftr, shuffled_X_msg, shuffled_X_code, shuffled_Y = X_ftr, X_msg, X_code, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]    

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for _ in range(num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X_ftr = shuffled_X_ftr[indexes]
        mini_batch_X_msg, mini_batch_X_code = shuffled_X_msg[indexes], shuffled_X_code[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
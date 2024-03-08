import os, torch, time
from jit_DExtended_model import DeepJITExtended
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from jit_utils import *

def evaluation_model(data, params):
    cc2ftr, code_loader, dict_msg, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, params.project)
    params.vocab_code = len(dict_code)
    params.vocab_msg = len(dict_msg)
    params.class_num = 1
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.feature_size = cc2ftr[0].shape[0]
    params.embedding_ftr = cc2ftr.shape[1]

    # create and train the defect model
    model = DeepJITExtended(args=params).to(device=params.device)
    model.load_state_dict(torch.load(params.load_model, map_location=params.device))

    model.eval()
    with torch.no_grad():
        all_predict, all_label = [], []

        # Record the start time
        start_time = time.time()

        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            cc2ftr = batch["feature"].to(params.device)
            
            # Forward
            predict = model(cc2ftr, message, code)

            all_predict += (predict.cpu().detach().numpy().tolist())
            all_label += (labels.cpu().detach().numpy().tolist())

        # Record the end time
        end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)

    ram = get_ram_usage()

    vram = get_vram_usage()

    # Call the function to write the content to the file
    log_auc(params.auc, params.project, auc_score)
    log_testing_time(params.testing_time, params.project, elapsed_time, "DeepJIT Extend")
    log_ram(params.ram, params.project, ram, "DeepJIT Extend")
    log_vram(params.vram, params.project, vram, "DeepJIT Extend")

    print('Test data -- AUC score:', auc_score)

    df = pd.DataFrame({'label': all_label, 'pred': all_predict})
    if os.path.isdir('./pred_scores/') is False:
        os.makedirs('./pred_scores/')
    df.to_csv('./pred_scores/test_com_' + params.project + '.csv', index=False, sep=',')
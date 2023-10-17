import os, datetime, torch
from jit_DExtended_model import DeepJITExtended
import pandas as pd
from sklearn.metrics import roc_auc_score

def write_to_file(file_path, content):
    with open(file_path, 'a+') as file:
        file.write(content + '\n')

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
        for batch in code_loader:
            # Extract data from DataLoader
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            cc2ftr = batch["feature"].to(params.device)
            
            # Forward
            predict = model(cc2ftr, message, code)

            all_predict += (predict.cpu().detach().numpy().tolist())
            all_label += (labels.cpu().detach().numpy().tolist())

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)

    # Call the function to write the content to the file
    write_to_file("auc.txt", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} - {params.project} - {auc_score}")

    print('Test data -- AUC score:', auc_score)

    df = pd.DataFrame({'label': all_label, 'pred': all_predict})
    if os.path.isdir('./pred_scores/') is False:
        os.makedirs('./pred_scores/')
    df.to_csv('./pred_scores/test_com_' + params.project + '.csv', index=False, sep=',')
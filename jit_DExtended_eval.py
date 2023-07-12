from jit_DExtended_model import DeepJITExtended
import torch 
from tqdm import tqdm
import os, datetime
import os, datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def evaluation_model(data, params):
    cc2ftr, code_loader, dict_msg, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)
    params.vocab_msg = len(dict_msg)
    params.class_num = 1
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.feature_size = cc2ftr[0].shape[0]

    # create and train the defect model
    model = DeepJITExtended(args=params).to(device=params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)

    model.eval()
    with torch.no_grad():
        all_predict, all_label = [], []
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            cc2ftr = batch["feature"].to(params.device)
            
            # Forward
            predict = model(cc2ftr, code, message)

            all_predict += (predict.cpu().detach().numpy().tolist())
            all_label += (labels.cpu().detach().numpy().tolist())

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)

    df = pd.DataFrame({'label': all_label, 'pred': all_predict})
    if os.path.isdir('./pred_scores/') is False:
        os.makedirs('./pred_scores/')
    df.to_csv('./pred_scores/test_com_' + params.project + '.csv', index=False, sep=',')
    print('Test data -- AUC score:', auc_score)

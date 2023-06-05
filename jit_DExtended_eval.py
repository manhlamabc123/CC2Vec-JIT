from jit_DExtended_model import DeepJITExtended
from jit_utils import mini_batches_DExtended
from sklearn.metrics import roc_auc_score    
import torch 
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd

def evaluation_model(data, params):
    cc2ftr, pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_DExtended(X_ftr=cc2ftr, X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.class_num = 1 if len(labels.shape) == 1 else labels.shape[1]
    params.embedding_ftr = cc2ftr.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = [], []
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, label = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    print(len(all_predict), len(all_label))
    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)

    # convert probabilities to binary predictions
    y_pred = [int(p >= 0.5) for p in all_predict]
    target_names = ['Clean', 'Defect']
    report = classification_report(all_label, y_pred, target_names=target_names, output_dict=True)
    # create DataFrame from report
    df = pd.DataFrame(report).transpose()

    # write DataFrame to CSV file
    df.to_csv('cc2vec_codeBERT_mean.csv')
    print('Test data -- AUC score:', auc_score)
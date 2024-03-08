import argparse, torch, random, os
from jit_DExtended_eval import evaluation_model
from jit_DExtended_train import train_model
from jit_DExtended_preprocess import preprocess_data
import numpy as np

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	#torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True
     
seed_torch()

def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-project', type=str, default='openstack', help='name of the dataset')

    parser.add_argument('-train', action='store_true', help='training DeepJIT model')  

    parser.add_argument('-train_data', type=str, help='the directory of our training data')
    parser.add_argument('-train_data_cc2ftr', type=str, help='the directory of our training data with cc2ftr')
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')
    parser.add_argument('-pred_data_cc2ftr', type=str, help='the directory of our testing data with cc2ftr')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=512, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training DeepJIT')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot_dextend', help='where to save the snapshot')
    parser.add_argument('-auc', type=str, default='', help='')
    parser.add_argument('-testing_time', type=str, default='', help='')
    parser.add_argument('-training_time', type=str, default='', help='')
    parser.add_argument('-ram', type=str, default='', help='')
    parser.add_argument('-vram', type=str, default='', help='')

    # CUDA
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params.train is True:
        data = preprocess_data(params)
        train_model(data=data, params=params)        
    elif params.predict is True:
        params.batch_size = 1
        data = preprocess_data(params)
        evaluation_model(data=data, params=params)
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from transformers import RobertaModel

# The HAN model
class HierachicalRNN(nn.Module):
    def __init__(self, args):
        super(HierachicalRNN, self).__init__()
        self.vocab_size = args.vocab_code
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.cls = args.class_num

        self.dropout = nn.Dropout(args.dropout_keep_prob)
    
        self.codeBERT = RobertaModel.from_pretrained("microsoft/codebert-base")

        # for param in self.codeBERT.base_model.parameters():
        #     param.requires_grad = False

        # standard neural network layer
        self.standard_nn_layer = nn.Linear(self.embed_size * 2, self.embed_size)

        # neural network tensor
        self.W_nn_tensor_one = nn.Linear(self.embed_size, self.embed_size)
        self.W_nn_tensor_two = nn.Linear(self.embed_size, self.embed_size)
        self.V_nn_tensor = nn.Linear(self.embed_size * 2, 2)

        # Hidden layers before putting to the output layer
        self.fc1 = nn.Linear(3 * self.embed_size + 4, 2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, self.cls)
        self.sigmoid = nn.Sigmoid()

    def forward_code(self, x):
        outputs = self.codeBERT(x)
        return outputs[0]

    def forward(self, added_code, removed_code):
        x_added_code = self.forward_code(x=added_code)
        x_removed_code = self.forward_code(x=removed_code)
        
        x_added_code= x_added_code[:, 0]
        x_removed_code = x_removed_code[:, 0]
        
        subtract = self.subtraction(added_code=x_added_code, removed_code=x_removed_code)
        multiple = self.multiplication(added_code=x_added_code, removed_code=x_removed_code)
        cos = self.cosine_similarity(added_code=x_added_code, removed_code=x_removed_code)
        euc = self.euclidean_similarity(added_code=x_added_code, removed_code=x_removed_code)
        nn = self.standard_neural_network_layer(added_code=x_added_code, removed_code=x_removed_code)
        ntn = self.neural_network_tensor_layer(added_code=x_added_code, removed_code=x_removed_code)

        x_diff_code = torch.cat((subtract, multiple, cos, euc, nn, ntn), dim=1)
        x_diff_code = self.dropout(x_diff_code)

        out = self.fc1(x_diff_code)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

    def forward_commit_embeds_diff(self, added_code, removed_code):
        x_added_code = self.forward_code(x=added_code)
        x_removed_code = self.forward_code(x=removed_code)

        x_added_code= x_added_code[:, 0]
        x_removed_code = x_removed_code[:, 0]

        subtract = self.subtraction(added_code=x_added_code, removed_code=x_removed_code)
        multiple = self.multiplication(added_code=x_added_code, removed_code=x_removed_code)
        cos = self.cosine_similarity(added_code=x_added_code, removed_code=x_removed_code)
        euc = self.euclidean_similarity(added_code=x_added_code, removed_code=x_removed_code)
        nn = self.standard_neural_network_layer(added_code=x_added_code, removed_code=x_removed_code)
        ntn = self.neural_network_tensor_layer(added_code=x_added_code, removed_code=x_removed_code)

        return torch.cat((subtract, multiple, cos, euc, nn, ntn), dim=1)

    def forward_commit_embeds(self, added_code, removed_code, hid_state_hunk, hid_state_sent, hid_state_word):
        hid_state = (hid_state_hunk, hid_state_sent, hid_state_word)

        x_added_code = self.forward_code(x=added_code, hid_state=hid_state)
        x_removed_code = self.forward_code(x=removed_code, hid_state=hid_state)

        x_added_code = x_added_code.view(self.batch_size, self.embed_size)
        x_removed_code = x_removed_code.view(self.batch_size, self.embed_size)

        return torch.cat((x_added_code, x_removed_code), dim=1)

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code

    def cosine_similarity(self, added_code, removed_code):
        cosine = nn.CosineSimilarity(eps=1e-6)
        return cosine(added_code, removed_code).view(added_code.shape[0], 1)

    def euclidean_similarity(self, added_code, removed_code):
        euclidean = nn.PairwiseDistance(p=2)
        return euclidean(added_code, removed_code).view(added_code.shape[0], 1)

    def standard_neural_network_layer(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=1)
        output = self.standard_nn_layer(concat)
        output = F.relu(output)
        return output

    def neural_network_tensor_layer(self, added_code, removed_code):
        output_one = self.W_nn_tensor_one(removed_code)
        output_one = torch.mul(output_one, added_code)
        output_one = torch.sum(output_one, dim=1).view(added_code.shape[0], 1)

        output_two = self.W_nn_tensor_two(removed_code)
        output_two = torch.mul(output_two, added_code)
        output_two = torch.sum(output_two, dim=1).view(added_code.shape[0], 1)

        W_output = torch.cat((output_one, output_two), dim=1)
        code = torch.cat((removed_code, added_code), dim=1)
        V_output = self.V_nn_tensor(code)
        return F.relu(W_output + V_output)
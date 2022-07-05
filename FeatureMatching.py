import torch.nn as nn
import torch
from GetDataset_Each_Feature import datasets_13_features
import os
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader, Subset
from random import sample
import math
import numpy as np



"""
Difference between this model and the former one is that "ModelForTemporalEmbeddings" only keeps the lstm lists 
for 13 features to get the embeddings of each feature
"""
class ModelForTemporalEmbeddings(nn.Module):
    def __init__(self, feature_num, hidden_dim_lstm, batch_size):
        super(ModelForTemporalEmbeddings, self).__init__()
        self.feature_num = feature_num
        self.hidden_dim_lstm = hidden_dim_lstm
        self.batch_size = batch_size
        self.lstm_list = []
        self.initial_hidden_cell_list = []
        for cur_feature in range(self.feature_num):
            self.lstm_list.append(nn.LSTM(2, self.hidden_dim_lstm).cuda())  # "2" is because input contains 1) the value for
            # the corresponding feature 2) mask for that value, 0 for missing and 1 for exist
            self.initial_hidden_cell_list.append((torch.randn(1, self.batch_size, self.hidden_dim_lstm).cuda(),
                                                  torch.randn(1, self.batch_size, self.hidden_dim_lstm).cuda()))

    def forward(self, input_data):
        # input_data: 13-length list, each element in this list is a packed_sequence
        features_embedding_list = []  # each element in this list trained from lstm has shape:
        # [1,self.batch_size, self.hidden_dim_lstm]
        # get each temporal feature's embeddings
        for cur_lstm in range(self.feature_num):
            cur_output, (cur_final_hidden_state, cur_final_cell_state) = self.lstm_list[cur_lstm](input_data[cur_lstm], self.initial_hidden_cell_list[cur_lstm])
            features_embedding_list.append(cur_final_hidden_state)
        return features_embedding_list # length: self.feature_num, each element has shape: [1, batch_size, hidden_dim_lstm]


def my_collate_for_feature_matching(batch):
    input_lstm_batch = [torch.cat((item[1], item[2]), 1) for item in batch]
    input_lstm_batch = pack_sequence(input_lstm_batch, enforce_sorted=False)
    return input_lstm_batch


"""
func Get2DataSets_FEM:
return XA, XB both are matrixes with size: 15000 x (num_features x embedding_size)
the first (num_matched_features x embedding_size) cols represent matched features
the left cols represent unmatched ones

total admissions in original dataset: 35623

"""


def get_2_FEM(model, embedding_size, num_features, num_matched_features=3, batch_size=10):
    datasets_13_features_A = []  # length: 15,000
    datasets_13_features_B = []  # length: 15,000
    dataloaders_13_features_A = []
    dataloaders_13_features_B = []
    for f_id in range(num_features):
        datasets_13_features_A.append(Subset(datasets_13_features[f_id], list(range(15000))))
        datasets_13_features_B.append(Subset(datasets_13_features[f_id], list(range(15000, 30000))))
        dataloaders_13_features_A.append(DataLoader(datasets_13_features_A[f_id], batch_size=batch_size,
                                                   collate_fn=my_collate_for_feature_matching, drop_last=True))
        dataloaders_13_features_B.append(DataLoader(datasets_13_features_B[f_id], batch_size=batch_size,
                                                   collate_fn=my_collate_for_feature_matching, drop_last=True))
    f_id_list = list(range(1, num_features+1))
    matched_features_id_list = sample(f_id_list, num_matched_features)  # eg: matched_features: [1, 8, 5]
    for mf_id in range(num_matched_features):
        f_id_list.remove(matched_features_id_list[mf_id])
    unmatched_features_id_list = f_id_list
    x_a = np.zeros((15000, num_features*embedding_size))
    x_b = np.zeros((15000, num_features*embedding_size))  # x_a & x_b are feature embeddings matrix for 2 datasets
    # get x_a
    for t, data in enumerate(zip(dataloaders_13_features_A[0], dataloaders_13_features_A[1], dataloaders_13_features_A[2],
                                 dataloaders_13_features_A[3], dataloaders_13_features_A[4], dataloaders_13_features_A[5],
                                 dataloaders_13_features_A[6], dataloaders_13_features_A[7], dataloaders_13_features_A[8],
                                 dataloaders_13_features_A[9], dataloaders_13_features_A[10], dataloaders_13_features_A[11],
                                 dataloaders_13_features_A[12])):
        embedding_list = model(data) # embedding_list is a "num_features"-length list,
        # with each element has shape: [1, batch_size, hidden_dim_lstm]
        for mf_id in range(num_matched_features):
            for b_id in range(batch_size):
                x_a[t*batch_size + b_id][mf_id*embedding_size:(mf_id+1)*embedding_size] = embedding_list[matched_features_id_list[mf_id]-1][0][b_id].detach().numpy()
        for umf_id in range(num_matched_features, num_features):
            for b_id in range(batch_size):
                x_a[t*batch_size + b_id][umf_id*embedding_size:(umf_id+1)*embedding_size] = embedding_list[unmatched_features_id_list[umf_id-num_matched_features]-1][0][b_id].detach().numpy()

    # get x_b
    for t, data in enumerate(zip(dataloaders_13_features_B[0], dataloaders_13_features_B[1], dataloaders_13_features_B[2],
                                 dataloaders_13_features_B[3], dataloaders_13_features_B[4], dataloaders_13_features_B[5],
                                 dataloaders_13_features_B[6], dataloaders_13_features_B[7], dataloaders_13_features_B[8],
                                 dataloaders_13_features_B[9], dataloaders_13_features_B[10], dataloaders_13_features_B[11],
                                 dataloaders_13_features_B[12])):
        embedding_list = model(data) # embedding_list is a "num_features"-length list,
        # with each element has shape: [1, batch_size, hidden_dim_lstm]
        for mf_id in range(num_matched_features):
            for b_id in range(batch_size):
                x_b[t*batch_size + b_id][mf_id*embedding_size:(mf_id+1)*embedding_size] = embedding_list[matched_features_id_list[mf_id]-1][0][b_id].detach().numpy()
        for umf_id in range(num_matched_features, num_features):
            for b_id in range(batch_size):
                x_b[t*batch_size + b_id][umf_id*embedding_size:(umf_id+1)*embedding_size] = embedding_list[unmatched_features_id_list[umf_id-num_matched_features]-1][0][b_id].detach().numpy()

    return x_a, x_b


def eu_Dist(map_fea_emb, unmap_fea_emb, embedding_size):  #map_fea_emb & unmap_fea_emb are numpy arrays with length "embedding_size"
    eu_dist = 0
    for i in range(embedding_size):
        eu_dist += (map_fea_emb[i] - unmap_fea_emb[i]) ** 2
    eu_dist = math.sqrt(eu_dist)
    return eu_dist


def get_2_DFM(num_matched_features, num_features, x_a, x_b, embedding_size):
    rows = num_matched_features
    cols = num_features-num_matched_features
    dfm_a = np.zeros((rows, cols))
    dfm_b = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            sum_eu_dist_a = 0
            sum_eu_dist_b = 0
            for i in range(15000):
                sum_eu_dist_a += eu_Dist(x_a[i][r*embedding_size : (r+1)*embedding_size], x_a[i][(c+num_matched_features)*embedding_size : (c+num_matched_features+1)*embedding_size])
                sum_eu_dist_b += eu_Dist(x_b[i][r*embedding_size : (r+1)*embedding_size], x_b[i][(c+num_matched_features)*embedding_size : (c+num_matched_features+1)*embedding_size])
            dfm_a[r][c] = sum_eu_dist_a / 15000
            dfm_b[r][c] = sum_eu_dist_b / 15000

    return dfm_a, dfm_b


feature_num = 13
best_config = {"hidden_dim_lstm": 6,
               "batch_size": 15}
final_model = ModelForTemporalEmbeddings(feature_num, best_config["hidden_dim_lstm"], best_config["batch_size"])
final_model.load_state_dict(torch.load('./best_sep_feature_model_weights.pth'), strict=False)


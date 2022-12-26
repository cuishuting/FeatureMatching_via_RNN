import torch
import torch.nn as nn
from GetDataset_All import OrgMIMICIIIDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import pairwise
from matching.games import HospitalResident
import pandas as pd



def Matching_via_HRM(C_X1_train, C_X2_train, P_x1_O_to_R, num_mapped_axis):  # in this case here the small feature sized database is X1, so we need to treat it as hospital and there will be capacities on it.
    ####### ----------  X1 train ------------- ##########

    true_features_pref_X1_train = {}
    cross_recon_features_pref_X1_train = {}
    capacities_X1_train = {}

    for i in range(C_X1_train.shape[0]):  # C_X1_train.shape[0]: number of unmapped features in dataset_1
        sorted_index = np.argsort(-C_X1_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X1_train["R" + str(i + 1)] = sorted_col_index
        capacities_X1_train["R" + str(i + 1)] = 1

    for j in range(C_X1_train.shape[1]): # C_X1_train.shape[1]:  number of unmapped features in dataset_2
        sorted_index = np.argsort(-C_X1_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X1_train["C" + str(j + 1)] = sorted_col_index

    game_X1_train = HospitalResident.create_from_dictionaries(cross_recon_features_pref_X1_train,
                                                              true_features_pref_X1_train,
                                                              capacities_X1_train)

    ####### ----------  X2 train ------------- ##########
    true_features_pref_X2_train = {}
    cross_recon_features_pref_X2_train = {}
    capacities_X2_train = {}

    for i in range(C_X2_train.shape[0]):  # C_X2_train.shape[0]: number of unmapped features in dataset_2
        sorted_index = np.argsort(-C_X2_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X2_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X2_train.shape[1]):  # C_X2_train.shape[1]: number of unmapped features in dataset_1
        sorted_index = np.argsort(-C_X2_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X2_train["C" + str(j + 1)] = sorted_col_index
        capacities_X2_train["C" + str(j + 1)] = 1

    game_X2_train = HospitalResident.create_from_dictionaries(true_features_pref_X2_train,
                                                              cross_recon_features_pref_X2_train,
                                                              capacities_X2_train)

       ######   ------------  Final matching -----------   ##########

    print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    print(matching_x1_train)

    print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    print(matching_x2_train)
    x1_train_y = [int(str(v[0])[1:]) if v else None for v in matching_x1_train.values()]
    x2_train_y = [int(str(v[0])[1:]) if v else None for v in matching_x2_train.values()]

    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    # shape: [num_unmapped_features_in_d1, num_unmapped_features_in_d2]
    matching_x2_train_matrix = np.zeros(np.transpose(C_X2_train).shape)
    # shape: [num_unmapped_features_in_d1, num_unmapped_features_in_d2]

    for i in range(matching_x1_train_matrix.shape[0]):  # number of unmapped features in d_1
        if x1_train_y[i] is not None:
            matching_x1_train_matrix[i, x1_train_y[i] - 1] = 1

    for i in range(matching_x2_train_matrix.shape[0]):  # number of unmapped features in d_1
        if x2_train_y[i] is not None:
            matching_x2_train_matrix[i, x2_train_y[i] - 1] = 1
    num_correct_from_x1 = 0
    num_correct_from_x2 = 0
    for i in range(P_x1_O_to_R.shape[0]):  # number of unmapped features in d_1
        if np.all(P_x1_O_to_R[i] == matching_x1_train_matrix[i]):
            num_correct_from_x1 = num_correct_from_x1 + 1
        if np.all(P_x1_O_to_R[i] == matching_x2_train_matrix[i]):
            num_correct_from_x2 = num_correct_from_x2 + 1

    return num_correct_from_x1, num_correct_from_x2, matching_x1_train_matrix, matching_x2_train_matrix


"""create a new model only containing the LSTM lists part of the original one to extract the hidden states as each feature's embeddings"""
class SepLSTMs(nn.Module):
    def __init__(self, batch_size, seq_len, temporal_feature_num, hidden_size_lstm):
        super(SepLSTMs, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_temporal = temporal_feature_num
        self.hidden_size_lstm = hidden_size_lstm
        # n lstms for n temporal features
        self.lstm_list = nn.ModuleList([nn.LSTMCell(2, self.hidden_size_lstm) for f in range(self.num_temporal)])

    def forward(self, input_data):
        sample_temporal_features = input_data[1]  # shape: [batch_size, 24, 12]
        sample_temporal_masks = input_data[2]  # shape: [batch_size, 24, 12]
        h_list = []  # hidden state list for 12 lstms
        c_list = []  # cell state list for 12 lstms

        # get final hidden state for each temporal feature: h_list[f] for temporal feature f
        for f, lstm in enumerate(self.lstm_list):
            h_list.append(torch.randn((self.batch_size, self.hidden_size_lstm)))
            c_list.append(torch.randn((self.batch_size, self.hidden_size_lstm)))
            # h_list.append(torch.randn((self.batch_size, self.hidden_size_lstm), generator=torch.Generator().manual_seed(2147483647)))
            # c_list.append(torch.randn((self.batch_size, self.hidden_size_lstm), generator=torch.Generator().manual_seed(2147483647)))
            # h_list.append(torch.zeros((self.batch_size, self.hidden_size_lstm)))
            # c_list.append(torch.zeros((self.batch_size, self.hidden_size_lstm)))
            cur_temporal_f = sample_temporal_features[:, :, f].reshape(self.batch_size, self.seq_len, 1)  # shape: [batch_size, 24, 1]
            cur_temporal_m = sample_temporal_masks[:, :, f].reshape(self.batch_size, self.seq_len, 1)  # shape: [batch_size, 24, 1]
            cur_input_lstm = torch.cat((cur_temporal_f, cur_temporal_m), 2)  # shape: [batch_size, 24, 2]
            for t in range(self.seq_len):
                cur_input = cur_input_lstm[:, t, :]  # shape: [batch_size, 2(input_size)]
                h_list[f], c_list[f] = self.lstm_list[f](cur_input, (h_list[f], c_list[f]))  # h_list[f]'s shape: [batch_size, hidden_size_lstm]
        return h_list  # len(h_list) equals to the number of temporal features, h_list[f] shape: [batch_size, hidden_size_lstm]


"""Get Embeddings for MV"""
"""'hidden_dim_lstm': 6, batch_size: 20, seq_len:24, temporal_feature_num: 82"""
data_dir_MV = "/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/test_imputed_MV_1_24.npz"
dataset_MV = OrgMIMICIIIDataset(data_dir_MV, "MV_test")
dataloader_MV = DataLoader(dataset_MV, batch_size=20, shuffle=True, drop_last=True)
batch_size_MV = 20
seq_len_MV = 24
temporal_feature_num_MV = 82
hidden_dim_lstm_MV = 6
model_MV_get_emb_only = SepLSTMs(batch_size_MV, seq_len_MV, temporal_feature_num_MV, hidden_dim_lstm_MV)
model_MV_get_emb_only.load_state_dict(torch.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Trained_Models_State/best_model_weights_MV_test.pth",
                                                 map_location=torch.device('cpu')), strict=False)
emb_matrix_MV = np.zeros((len(dataset_MV), temporal_feature_num_MV*hidden_dim_lstm_MV))

model_MV_get_emb_only.eval()
for i, data in enumerate(dataloader_MV):
    cur_emb_list_all_features = model_MV_get_emb_only(data)
    temp_result = np.concatenate([cur_feature.detach().numpy() for cur_feature in cur_emb_list_all_features], axis=1)
    emb_matrix_MV[i*batch_size_MV:(i+1)*batch_size_MV][:] = temp_result

"""Get Embeddings for CV"""
"""'hidden_dim_lstm': 4， 'batch_size': 10， temporal_feature_num: 101"""
data_dir_CV = "/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/test_imputed_CV_1_24.npz"
dataset_CV = OrgMIMICIIIDataset(data_dir_CV, "CV_test")
dataloader_CV = DataLoader(dataset_CV, batch_size=10, shuffle=True, drop_last=True)
batch_size_CV = 10
seq_len_CV = 24
temporal_feature_num_CV = 101
hidden_dim_lstm_CV = 4
model_CV_get_emb_only = SepLSTMs(batch_size_CV, seq_len_CV, temporal_feature_num_CV, hidden_dim_lstm_CV)
model_CV_get_emb_only.load_state_dict(torch.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Trained_Models_State/best_model_weights_CV_test.pth",
                                                 map_location=torch.device('cpu')), strict=False)
emb_matrix_CV = np.zeros((len(dataset_CV), temporal_feature_num_CV*hidden_dim_lstm_CV))
model_CV_get_emb_only.eval()
for i, data in enumerate(dataloader_CV):
    cur_emb_list_all_features = model_CV_get_emb_only(data)
    temp_result = np.concatenate([cur_feature.detach().numpy() for cur_feature in cur_emb_list_all_features], axis=1)
    emb_matrix_CV[i*batch_size_CV:(i+1)*batch_size_CV][:] = temp_result

"""calculate the euclidean-distance matrix for 2 eras between unmapped features and all the mapped features in the same era"""
"""MV"""
ump_f_MV = 46
mp_f_MV = 36
Eu_Dis_Matrix_MV = np.zeros((ump_f_MV, mp_f_MV))
for i in range(ump_f_MV):
    cur_ump_f_emb = emb_matrix_MV[:][(mp_f_MV+i)*hidden_dim_lstm_MV: (mp_f_MV+i+1)*hidden_dim_lstm_MV]
    for j in range(mp_f_MV):
        cur_mp_f_emb = emb_matrix_MV[:][j*hidden_dim_lstm_MV: (j+1)*hidden_dim_lstm_MV]
        Eu_Dis_Matrix_MV[i][j] = np.average(np.linalg.norm(cur_ump_f_emb - cur_mp_f_emb, ord=2, axis=1))

"""CV"""
ump_f_CV = 65
mp_f_CV = 36
Eu_Dis_Matrix_CV = np.zeros((ump_f_CV, mp_f_CV))
for i in range(ump_f_CV):
    cur_ump_f_emb = emb_matrix_CV[:][(mp_f_CV+i)*hidden_dim_lstm_CV: (mp_f_CV+i+1)*hidden_dim_lstm_CV]
    for j in range(mp_f_CV):
        cur_mp_f_emb = emb_matrix_CV[:][j*hidden_dim_lstm_CV: (j+1)*hidden_dim_lstm_CV]
        Eu_Dis_Matrix_CV[i][j] = np.average(np.linalg.norm(cur_ump_f_emb - cur_mp_f_emb, ord=2, axis=1))


"""Get cosine similarity matrix between unmapped features in MV and CV"""
cos_similarity_matrix_X1_to_X2 = pairwise.cosine_similarity(Eu_Dis_Matrix_MV, Eu_Dis_Matrix_CV, dense_output=True)
cos_similarity_matrix_X2_to_X1 = pairwise.cosine_similarity(Eu_Dis_Matrix_CV, Eu_Dis_Matrix_MV, dense_output=True)

"""Get P_x1: true permutation matrix between only unmatched features"""
P_x1 = np.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Permutation_Matrix_both_mp_ump.npy")

correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(cos_similarity_matrix_X1_to_X2, cos_similarity_matrix_X2_to_X1, P_x1[mp_f_MV:, mp_f_CV:], mp_f_CV)
temp_inf_x1 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS'])
temp_inf_x2 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS'])

data_mv_no_tw = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tab_MV_no_tw_no_fake_chart.csv")
data_cv_no_tw = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tab_CV_no_tw_no_fake_chart.csv")

# column names in dataset_MV
reordered_column_names_orig = data_mv_no_tw.columns
# column names in dataset_CV
reordered_column_names_r = data_cv_no_tw.columns
mapped_features = list(set(reordered_column_names_orig).intersection(set(reordered_column_names_r)))

for i in range(x1_match_matrix_test.shape[0]):
    matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
    temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]

for i in range(x2_match_matrix_test.shape[0]):
    matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
    temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
    # temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
    temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]

TP_x1 = 0
FP_x1 = 0
TN_x1 = 0
FN_x1 = 0
P_x1_only_unmap = P_x1[mp_f_MV:, mp_f_CV:]
for i in range(P_x1_only_unmap.shape[0]):
    for j in range(P_x1_only_unmap.shape[1]):
        if (P_x1_only_unmap[i, j] == 1) & (x1_match_matrix_test[i, j] == 1):
            TP_x1 = TP_x1 + 1
        elif (P_x1_only_unmap[i, j] == 1) & (x1_match_matrix_test[i, j] == 0):
            FN_x1 = FN_x1 + 1
        elif (P_x1_only_unmap[i, j] == 0) & (x1_match_matrix_test[i, j] == 0):
            TN_x1 = TN_x1 + 1
        elif (P_x1_only_unmap[i, j] == 0) & (x1_match_matrix_test[i, j] == 1):
            FP_x1 = FP_x1 + 1

TP_x2 = 0
FP_x2 = 0
TN_x2 = 0
FN_x2 = 0
for i in range(P_x1_only_unmap.shape[0]):
    for j in range(P_x1_only_unmap.shape[1]):
        if (P_x1_only_unmap[i, j] == 1) & (x2_match_matrix_test[i, j] == 1):
            TP_x2 = TP_x2 + 1
        elif (P_x1_only_unmap[i, j] == 1) & (x2_match_matrix_test[i, j] == 0):
            FN_x2 = FN_x2 + 1
        elif (P_x1_only_unmap[i, j] == 0) & (x2_match_matrix_test[i, j] == 0):
            TN_x2 = TN_x2 + 1
        elif (P_x1_only_unmap[i, j] == 0) & (x2_match_matrix_test[i, j] == 1):
            FP_x2 = FP_x2 + 1
F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)
print("Sim cor F values ", F1_fromx1, F1_fromx2)


"""Get GS_match_result_of_method2"""
item_id_dbsource = pd.read_csv('/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Org_Data/d_items_chartevents.csv')
selected_itemid_dbsource_for_ump_x1 = item_id_dbsource[["itemid", "label"]].astype({"itemid": "int64"})
selected_itemid_dbsource_for_ump_x1.rename(columns={"itemid": "ump_feature_in_X1", "label": "ump_feature_in_X1_label"}, inplace=True)
temp_inf_x1 = temp_inf_x1.astype({"ump_feature_in_X1": "int64"})
new_inf_x1 = temp_inf_x1.merge(selected_itemid_dbsource_for_ump_x1, how="inner", on="ump_feature_in_X1")

selected_itemid_dbsource_for_predicted_matched_byGS = item_id_dbsource[["itemid", "label"]].astype({"itemid": "int64"})
selected_itemid_dbsource_for_predicted_matched_byGS.rename(columns={"itemid": "match_byGS", "label": "match_byGS_label"}, inplace=True)
temp_inf_x2 = temp_inf_x2.astype({"match_byGS": "int64"})
new_inf_x2 = temp_inf_x2.merge(selected_itemid_dbsource_for_predicted_matched_byGS, how="inner", on="match_byGS")

matched_pairs_itemid_label = pd.concat([new_inf_x1[["ump_feature_in_X1", "ump_feature_in_X1_label", "match_byGS"]], new_inf_x2[["match_byGS_label"]]], axis=1)
# print(new_inf_x1)
# print(new_inf_x2)
# matched_pairs_itemid_label.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/CheckPoint_Files/GS_match_result.csv", index=False)
matched_pairs_itemid_label.rename(columns={"ump_feature_in_X1": "ump_feature_in_MV",
                                           "ump_feature_in_X1_label": "ump_feature_in_MV_label",
                                           "match_byGS": "match_byGS_CV",
                                           "match_byGS_label": "match_byGS_CV_label"}, inplace=True)
GoldStandard_match = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/CheckPoint_Files/GoldStandard_match.csv")
GoldStandard_match.rename(columns={"CV_itemids": "GoldStd_match_CV_itemids",
                                   "CV_labels": "GoldStd_match_CV_labels",
                                   "MV_itemids": "ump_feature_in_MV",
                                   "MV_labels": "GoldStd_match_MV_labels"}, inplace=True)
merge_GS_result_and_GoldStandard = matched_pairs_itemid_label.merge(GoldStandard_match[["ump_feature_in_MV", "GoldStd_match_CV_itemids", "GoldStd_match_CV_labels"]],
                                                             on="ump_feature_in_MV", how="left")
merge_GS_result_and_GoldStandard = merge_GS_result_and_GoldStandard[["ump_feature_in_MV",
                                                                     "ump_feature_in_MV_label",
                                                                     "match_byGS_CV",
                                                                     "match_byGS_CV_label",
                                                                     "GoldStd_match_CV_itemids",
                                                                     "GoldStd_match_CV_labels"]]
merge_GS_result_and_GoldStandard.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/match_result/GS_match_by_RNN.csv", index=False)



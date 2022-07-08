import torch
import torch.nn as nn
from random import sample
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import optim

# Only take the first 4,000 admissions' observations for checking the data performance


class BaseModelDataset(Dataset):
    def __init__(self, path):
        self.org_data = np.load(path, allow_pickle=True)
        self.static_feature_list = []
        self.mask_list = []
        self.temporal_feature_list = []
        self.target_list = []
        for i in range(4000):
            self.static_feature_list.append(torch.tensor(self.org_data["adm_features_all"][i]).float())
            # static_feature: torch.Size([5])
            self.mask_list.append(torch.tensor(self.org_data['X_t_mask'][i]).float())
            # mask: torch.Size([time_stamps, 13])
            self.temporal_feature_list.append(torch.tensor(self.org_data['X_t'][i]).float())
            # temporal_feature: torch.Size([time_stamps, 13])
            cur_icd9_labels = torch.tensor(self.org_data['y_icd9'][i])
            cur_y_mor = torch.tensor(self.org_data['y_mor'][i])
            self.target_list.append(torch.cat((cur_icd9_labels, cur_y_mor), 0))
            # target: torch.Size([21])

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        #  for each admission, randomly select one time stamp to extract the
        #  corresponding temporal feature and mask vector
        #  then concat them together with the admission's static variable
        cur_time_points_list = list(range(len(self.mask_list[idx])))
        selected_time_point = sample(cur_time_points_list, 1)[0]
        cur_temporal_feature = self.temporal_feature_list[idx][selected_time_point]
        cur_mask = self.mask_list[idx][selected_time_point]
        cur_static_feature = self.static_feature_list[idx]
        input = torch.cat((cur_temporal_feature, cur_mask, cur_static_feature), 0)
        target = self.target_list[idx]
        return input, target

# using the first 3000 admissions data for base model training
# using the next 1000 admissions data for base model testing

# The BaseLineModel is only a MLP model with
# input size: temporal+mask+static (31)
# output size: targets_num (21)
# intermediate layer size: 30 & 25
class BaseLineModel(nn.Module):
    def __init__(self, temporal_feature_num, static_feature_num, targets_num, dropout_p):
        super(BaseLineModel, self).__init__()
        self.input_dim = temporal_feature_num*2 + static_feature_num
        self.output_dim = targets_num
        self.dense1_size = 30
        self.dense2_size = 25
        self.dropout_p = dropout_p
        self.build()

    def build(self):
        self.input_layer = nn.Linear(self.input_dim, self.dense1_size)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(self.dense1_size, self.dense2_size)
        self.dense2 = nn.Linear(self.dense2_size, self.output_dim)
        self.dropout = nn.Dropout(p = self.dropout_p)

    def forward(self, input_data):
        # input_data shape: [31]
        temp_result = self.input_layer(input_data)
        temp_result = self.relu(temp_result)
        temp_result = self.dense1(temp_result)
        temp_result = self.relu(temp_result)
        temp_result = self.dense2(temp_result)
        predict = self.dropout(temp_result)
        return predict

    def Loss(self, predict, target):
        return nn.functional.binary_cross_entropy_with_logits(predict, target)


def TestProcess(trained_model, testloader):
    auroc = np.zeros(21)
    auprc = np.zeros(21)
    base_performance_for_auprc = np.zeros(21)
    predicts_for_auroc_auprc = np.zeros((21, 1000))
    targets_for_auroc_auprc = np.zeros((21, 1000))
    for t, (input_data, target) in enumerate(testloader):
        predict = trained_model(input_data)
        for i in range(21):
            predicts_for_auroc_auprc[i][t] = predict[0][i]
            targets_for_auroc_auprc[i][t] = target[0][i]

    for t_id in range(21):
        auroc[t_id] = roc_auc_score(targets_for_auroc_auprc[t_id], predicts_for_auroc_auprc[t_id])
        auprc[t_id] = average_precision_score(targets_for_auroc_auprc[t_id], predicts_for_auroc_auprc[t_id])
        base_performance_for_auprc[t_id] = np.sum(targets_for_auroc_auprc[t_id] == 1) / len(targets_for_auroc_auprc[t_id])
    print("21 targets' auroc:\n")
    print(auroc)
    print("base performance of auprc on 21 targets are:\n")
    print(base_performance_for_auprc)
    print("21 targets' auprc:\n")
    print(auprc)


def main():
    dataset = BaseModelDataset("./MIMIC_timeseries/24hours/series/normed-ep.npz")
    train_set, test_set = random_split(dataset, [3000, 1000], generator=torch.Generator().manual_seed(420))
    batch_size = 10
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set)
    temporal_feature_num = 13
    static_feature_num = 5
    targets_num = 21
    dropout_p = 0.2
    model = BaseLineModel(temporal_feature_num, static_feature_num, targets_num, dropout_p)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        print("cuda is available!\n")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.05)

    # Train Process
    for epoch in range(10):
        for t, (input_data, target) in enumerate(train_loader):
            input_data = input_data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            predict_list = model(input_data)
            loss = model.Loss(predict_list, target.float())
            loss.backward()
            optimizer.step()
            if (t+1) % 50 == 0:
                print("[%d, %d] loss: %.3f" % (epoch+1, t+1, loss))

    # Test Process
    TestProcess(model, test_loader)





main()

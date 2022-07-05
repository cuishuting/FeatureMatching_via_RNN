"""
transform all the orginal time series data and targets into tensors
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MIMICIII_Dataset(Dataset):
    def __init__(self, path, feature_dim):
        self.org_data = np.load(path, allow_pickle=True)
        self.input_list = []
        self.target_list = []
        if os.path.exists("DataListTensor.pt") and os.path.exists("TargetListTensor.pt"):
            self.input_list = torch.load("DataListTensor.pt")
            self.target_list = torch.load("TargetListTensor.pt")
        else:
            # for i in range(100):
            for i in range(len(self.org_data['X_t'])):
                if (i+1) % 1000 == 0:
                 print("currently transform: ", i+1)
                self.input_list.append(torch.tensor(self.org_data['X_t'][i]).float().view(-1, 1, feature_dim).cuda())
                self.target_list.append(torch.tensor(self.org_data['y_mor'][i]).cuda())
            torch.save(self.input_list, "DataListTensor.pt")
            torch.save(self.target_list, "TargetListTensor.pt")

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        sample_data = self.input_list[idx]
        sample_target = self.target_list[idx]
        return sample_data, sample_target


data_path = "MIMIC_timeseries/24hours/series/normed_ep_ratio.npz"
print("begin transform original data\n")
dataset = MIMICIII_Dataset(data_path, 12)
print("finish transforming original data\n")
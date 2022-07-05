# New dataset class giving "static variables + temporal feature data" for each admission
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class MIMICIII_Dataset_with_Static(Dataset):
    def __init__(self, path, feature_dim):
        self.org_data = np.load(path, allow_pickle=True)
        self.temporal_list = []
        self.target_list = []
        self.static_list = []
        if os.path.exists("DataListTensor.pt"):
            self.temporal_list = torch.load("DataListTensor.pt")
        else:
            for i in range(len(self.org_data['X_t'])):
                if (i + 1) % 1000 == 0:
                    print("currently transform: ", i + 1)
                self.temporal_list.append(torch.tensor(self.org_data['X_t'][i]).float().view(-1, 1, feature_dim).cuda())
            torch.save(self.temporal_list, "DataListTensor.pt")

        if os.path.exists("TargetListTensor.pt"):
            self.target_list = torch.load("TargetListTensor.pt")
        else:
            for i in range(len(self.org_data['X_t'])):
                if (i + 1) % 1000 == 0:
                    print("currently transform: ", i + 1)
                self.target_list.append(torch.tensor(self.org_data['y_mor'][i]).cuda())
            torch.save(self.target_list, "TargetListTensor.pt")

        if os.path.exists("StaticListTensor.pt"):
            self.static_list = torch.load("StaticListTensor.pt")
        else:
            for i in range(len(self.org_data['X_t'])):
                if (i + 1) % 1000 == 0:
                    print("currently transform: ", i + 1)
                self.static_list.append(torch.tensor(self.org_data["adm_features_all"][i]).float().cuda())
            torch.save(self.static_list, "StaticListTensor.pt")

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        sample_temporal_data = self.temporal_list[idx]
        sample_static_data = self.static_list[idx]
        sample_target = self.target_list[idx]
        return sample_static_data, sample_temporal_data, sample_target


data_path = "MIMIC_timeseries/24hours/series/normed_ep_ratio.npz"
print("begin transform original data\n")
dataset = MIMICIII_Dataset_with_Static(data_path, 12)
print("finish transforming original data\n")
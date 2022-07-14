import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import os


class MIMICIIIEachFeature(Dataset):
    def __init__(self, path, feature_id):
        self.org_data = np.load(path, allow_pickle=True)
        self.feature_id = feature_id
        self.static_feature_list = []
        self.temporal_feature_list = []
        self.mask_list = []  # mask has the shape with temporal features
        self.target_list = []  # target list has dim==21 : 20 for icd9 labels and 1 for mortality labels

        # load static variables

        if os.path.exists("./StaticListTensor_sampled_imputed.pt"):
            self.static_feature_list = torch.load("./StaticListTensor_sampled_imputed.pt")
            print("Successfully load static variables!\n")
        else:
            for s in range(len(self.org_data['adm_features_all'])):
                if (s + 1) % 3000 == 0:
                    print("currently transform (static feature): ", s + 1)
                self.static_feature_list.append(torch.tensor(self.org_data["adm_features_all"][s]).float().cuda())
            torch.save(self.static_feature_list, "StaticListTensor_sampled_imputed.pt")

        # load temporal features

        if os.path.exists("./TemporalFeatureTensor_sampled_imputed_12.pt"):
            self.temporal_feature_list = torch.load("./TemporalFeatureTensor_sampled_imputed_12.pt")
            print("Successfully load temporal features!\n")
        else:
            for t in range(len(self.org_data['ep_tdata'])):
                if (t + 1) % 3000 == 0:
                    print("currently transform (temporal feature): ", t + 1)

                # turn "nan" in temporal features into zero
                cur_temporal_feature = torch.tensor(self.org_data['ep_tdata'][t])
                cur_temporal_feature[cur_temporal_feature != cur_temporal_feature] = 0

                self.temporal_feature_list.append(cur_temporal_feature.float().cuda())
            torch.save(self.temporal_feature_list, "TemporalFeatureTensor_sampled_imputed_12.pt")

        # load mask
        if os.path.exists("./MaskTensor_12_sampled_imputed.pt"):
            self.mask_list = torch.load("./MaskTensor_12_sampled_imputed.pt")
            print("Successfully load mask!\n")
        else:
            for m in range(len(self.org_data['ep_tdata_masking'])):
                if (m + 1) % 3000 == 0:
                    print("currently transform (mask): ", m+1)
                self.mask_list.append(torch.tensor(self.org_data['ep_tdata_masking'][m]).cuda())

            torch.save(self.mask_list, "MaskTensor_12_sampled_imputed.pt")

        # load targets (dim == 21)

        if os.path.exists("./MultitaskTarget_sampled_imputed.pt"):
            self.target_list = torch.load("./MultitaskTarget_sampled_imputed.pt")
            print("Successfully load multi targets!\n")
        else:
            for i in range(len(self.org_data['y_icd9'])):
                if (i + 1) % 3000 == 0:
                    print("currently transform (y_mor + icd9_labels): ", i + 1)
                cur_icd9_labels = torch.tensor(self.org_data['y_icd9'][i])
                cur_y_mor = torch.tensor(self.org_data['y_mor'][i])
                self.target_list.append(torch.cat((cur_icd9_labels, cur_y_mor), 0).cuda())
            torch.save(self.target_list, "MultitaskTarget_sampled_imputed.pt")

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        sample_static_data = self.static_feature_list[idx]
        sample_temporal_data = self.temporal_feature_list[idx][:, self.feature_id]
        sample_mask_data = self.mask_list[idx][:, self.feature_id]
        sample_target = self.target_list[idx]
        return sample_static_data, sample_temporal_data, sample_mask_data, sample_target


data_dir = os.path.abspath("./MIMIC_timeseries/24hours/series/imputed-normed-ep_1_24.npz")
datasets_12_features = []
for f in range(12):
    datasets_12_features.append(MIMICIIIEachFeature(data_dir, f))
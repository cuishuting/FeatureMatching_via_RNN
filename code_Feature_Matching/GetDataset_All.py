"""
Difference between GetDataset_All.py & GetDataset_Each_Feature.py:
    GetDataset_Each_Feature.py: Return each temporal feature's separate values in dataset defining files
    GetDataset_All.py: Return the original form of temporal features, getting each feature's separate value in model training process
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class OrgMIMICIIIDataset(Dataset):
    def __init__(self, path, features_belongs):
        self.all_data = np.load(path, allow_pickle=True)
        self.static_features_list = []
        self.temporal_features_list = []
        self.temporal_masks_list = []
        self.targets_list = []
        self.num_data = len(self.all_data['y_mor'])
        # load static variables
        if os.path.exists("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/StaticFeatures_" + features_belongs + ".pt"):
            self.static_features_list = torch.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/StaticFeatures_" + features_belongs + ".pt")
            print("Successfully load static variables!\n")
        else:
            for s in range(self.num_data):
                if (s+1) % 3000 == 0:
                    print("currently transform (static feature): ", s + 1)
                cur_static_feature = torch.tensor(self.all_data["adm_features_all"][s])
                cur_static_feature[cur_static_feature != cur_static_feature] = 0
                self.static_features_list.append(cur_static_feature.float())

            torch.save(self.static_features_list, "/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/StaticFeatures_" + features_belongs + ".pt")

        # load temporal features
        if os.path.exists("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/TemporalFeatures_" + features_belongs + ".pt"):
            self.temporal_features_list = torch.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/TemporalFeatures_" + features_belongs + ".pt")
            print("Successfully load temporal features!\n")
        else:
            for t in range(self.num_data):
                if (t + 1) % 3000 == 0:
                    print("currently transform (temporal feature): ", t + 1)
                # turn "nan" in temporal features into zero
                cur_temporal_feature = torch.tensor(self.all_data['ep_tdata'][t])
                cur_temporal_feature[cur_temporal_feature != cur_temporal_feature] = 0
                self.temporal_features_list.append(cur_temporal_feature.float())

            torch.save(self.temporal_features_list, "/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/TemporalFeatures_" + features_belongs + ".pt")

        # load mask
        if os.path.exists("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/TemporalMasks_" + features_belongs + ".pt"):
            self.temporal_masks_list = torch.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/TemporalMasks_" + features_belongs + ".pt")
            print("Successfully load mask!\n")
        else:
            for m in range(self.num_data):
                if (m + 1) % 3000 == 0:
                    print("currently transform (mask): ", m + 1)
                self.temporal_masks_list.append(torch.tensor(self.all_data['ep_tdata_masking'][m]))

            torch.save(self.temporal_masks_list, "/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Train_Tensors/TemporalMasks_" + features_belongs + ".pt")

        # load targets: a) data['y_icd9'][6] (t7) b) data['y_mor'] (mortality label)
        if os.path.exists("/Targets_08_17.pt"):
            self.targets_list = torch.load("./Targets_08_17.pt")
            print("Successfully load targets!\n")
        else:
            for i in range(self.num_data):
                if (i + 1) % 3000 == 0:
                    print("currently transform targets: ", i + 1)
                cur_icd9_labels = torch.tensor(self.all_data["y_icd9"][i])
                cur_y_mor = torch.tensor(self.all_data["y_mor"][i])
                cur_target = torch.cat((cur_icd9_labels, cur_y_mor), 0)
                self.targets_list.append(cur_target)

            torch.save(self.targets_list, "Targets_08_17.pt")

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        sample_static_feature = self.static_features_list[idx]
        sample_temporal_feature = self.temporal_features_list[idx]  # [24, 12]
        sample_temporal_mask = self.temporal_masks_list[idx]  # [24, 12]
        sample_target = self.targets_list[idx]
        return sample_static_feature, sample_temporal_feature, sample_temporal_mask, sample_target





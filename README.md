# FeatureMatching_via_RNN
Because our goal is to get fingerprint for each temporal feature, we use seperate LSTMs to learn each time series data. We also add the prior knowledge from static variables and concatenate the temporal features’ embeddings and static embeddings to predict multi-targets. The reasonability of this structure is that we believe there is strong correlation between temporal features and the icd9 labels plus mortality labels. So do the static variables. The structure of the model is as follow.
![image](https://github.com/cuishuting/FeatureMatching_via_RNN/blob/main/IMG/model_structure_rnn_model.png)

Four baseline models' performance:
* AUPRC of four baseline models (the first line is the base AUPRC performance based on the positive rate in each target class)
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/AUPRC_compares.png)
* AUROC of four baseline models
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/AUROC_compares.png)
The four baseline models includes:
* **sklearn_LR_simple_input**:
  * **model**: 21 separate sklearn.linear_model.LogisticRegression models, each for one target
  * **training data**:
    * predictors = np.concatenate((static_feature, temporal_feature, temporal_mask), 1), shape: [adm_num, 29]
      * static_feature : shape: [adm_num, static_feature_dim(5)]
      * temporal_feature : shape: [adm_num, temporal_feature_dim(12)], variable "temporal_feature[i,j]" denotes admission i's first non-missing measurement for feature j, if admission i doesn't have non-missing measurement, then temporal_feature[i,j] == 0
      * temporal_mask : shape: [adm_num, temporal_feature_dim(12)], temporal_mask[i,j] == 1 if admission i has non-missing measurement for feature j, otherwise, temporal_mask[i,j] == 0
  * **labels**:
      * targets = np.concatenate((y_icd9, y_mor), 1), shape: [adm_num, 21]
        * y_icd9: shape: [adm_num, 20]
        * y_mor: shape: [adm_num, 1]

* **pytorch_LR_simple_input**:
  * **model**: pytorch multi-label binary classification problem
    * n_classes: 21 (y_icd9: 20, y_mor: 1)
    * n_labels for each class: 2 (0/1)
    * model structure: 2-layer MLP, the first layer's size is the same as the input dim; the output layer's size is the same as targets dim(21) without sigmoid activation function here, because we use BCEWithLogitsLoss as loss function, which combines a Sigmoid layer and the BCELoss in one single class
  * **training data**:
    * input_tensor = torch.cat((static_feature_tensor, temporal_feature_tensor, temporal_mask_tensor), 1), all of the three above variables are the same as those in the first model but the dtype is tensor.
  * **labels**:
    * targets_tensor = torch.tensor(targets) same as "targets" in the first model but in the tensor form

* **pytorch_LR_static_only**:
pytorch baseline model with only static features
  * **model**: the model structure is the same as the second model, but the input size for the first layer changed to static feature dim(5)
  * **input**: only the static variable
  * **targets**: still the 21 targets
* **pytorch_LR_temporal_cat_static_org_input**:
  * **model**: includes a LSTMCell structrue to get embeddings for temporal features, an one-layer MLP to get the prediction for 21 classes
  * **input**:
    * LSTMCell : original temporal features on each time stamp's with shape: [batch_size, temporal_feature_dim(12)]
    * one layer MLP: final hidden state from LSTMCell model concatenate with static features with shape: [batch_size, final_hidden_embedding_dim+static_dim]
  * **targets**: still the 21 targets

This repository includes files for:
* Different models to get the embeddings for temporal features (each model's structure is shown below):
  * GetDataset.py & main.py
  * GetDataset_static.py & main_static.py
  * GetDataset_Each_Feature.py & main_13_features.py
* Baseline models to see how time series data and static data are work for predicting multi-targets:
  * main_baseline.py
* Feature matching algorithm (not finish)
  * FeatureMatching.py
### GetDataset.py & main.py 
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/main_model.png)

### GetDataset_static.py & main_static.py
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/main_static_model.png)

### GetDataset_Each_Feature.py & main_13_features.py
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/Screen%20Shot%202022-07-08%20at%204.01.46%20PM.png)
* AUPRC of each feature on each target
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/AUPRC_main_model.png)
* AUROC of each feature on each target
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/auroc_main_model.png)
### main_baseline.py
![image](https://github.com/cuishuting/TimeSeries_Analysis/blob/main/IMG/baseline_model.png)

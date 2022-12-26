# FeatureMatching_via_RNN
Because our goal is to get fingerprint for each temporal feature, we use seperate LSTMs to learn each time series data. We also add the prior knowledge from static variables and concatenate the temporal features’ embeddings and static embeddings to predict multi-targets. The reasonability of this structure is that we believe there is strong correlation between temporal features and the icd9 labels plus mortality labels. So do the static variables. The structure of the model is as follow.
![image](https://github.com/cuishuting/FeatureMatching_via_RNN/blob/main/IMG/model_structure_rnn_model.png)
After hyperparameter tuning, we store the best models’(with least cross entropy loss on test set) state for both era. Then another model only containing the LSTMs parts will be warmstarted from the best models’ state. And again, feeding the processed data into this reference model once, we get the fingerprints for each temporal feature both mapped and unmapped.

After getting the fingerprints for both mapped and unmapped temporal features, we then apply the KMF module as shown below to get embeddings for each unmapped feature in 2 datasets and calculate cosine similarity between unmapped features in both datasets as the "preference list" for each unmapped feature in one dataset based on unmapped features' embeddings. Then Gale-Shapley algorithm is applied to get the predicted matched pairs for unmapped features in two datasets.
![image](https://github.com/cuishuting/FeatureMatching_via_RNN/blob/main/IMG/KMF_RNN.png)

Current match result through extracting final hidden states from LSTM models is shown below with F-1 score 0.108. More will update later to improve matching accuracy through RNN models.
![image](https://github.com/cuishuting/FeatureMatching_via_RNN/blob/main/IMG/match_result_rnn_1.png)
![image](https://github.com/cuishuting/FeatureMatching_via_RNN/blob/main/IMG/match_result_rnn_2.png)


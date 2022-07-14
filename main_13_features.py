import torch
import numpy as np
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split, DataLoader
from GetDataset_Each_Feature import datasets_12_features
from torch.nn.utils.rnn import pack_sequence
from sklearn.metrics import average_precision_score, roc_auc_score
import random
from functools import partial



# define 11 separate LSTM models out of the cross model class
# The only class needed to be defined is the later MLP class
# which will perform the concatenation of the embeddings from 11 lstms

class MIMICIIIWithSepFeatures(nn.Module):
    def __init__(self, feature_num, static_feature_dim, static_embeddings_dim, hidden_dim_lstm, dropout_p, batch_size):
        super(MIMICIIIWithSepFeatures, self).__init__()
        # feature_num: the number of features' types; feature_num separate lstms needed to be trained
        # hidden_dim_lstm: each lstm has the same hidden_size
        self.feature_num = feature_num
        self.static_feature_dim = static_feature_dim
        self.static_embeddings_dim = static_embeddings_dim
        self.hidden_dim_lstm = hidden_dim_lstm
        self.dropout_p = dropout_p
        self.batch_size = batch_size  # batch_size is used for packed sequence in lstm
        self.lstm_list = []
        self.initial_hidden_cell_list = []
        self.mlp_list = nn.ModuleList()  # these mlps are in the final part of the whole structure
        for cur_feature in range(self.feature_num):
            self.lstm_list.append(nn.LSTM(2, self.hidden_dim_lstm, batch_first=True).cuda())
            #  "2" is because input contains: 1) the value for the corresponding feature
            #  2) mask for that value, 0 for missing and 1 for exist
            self.initial_hidden_cell_list.append((torch.randn(1, self.batch_size, self.hidden_dim_lstm).cuda(),
                                                  torch.randn(1, self.batch_size, self.hidden_dim_lstm).cuda()))

        # Below is the structure aimed at getting the static variable's embeddings
        # (The result is a tensor with shape: (1, 10, 3), which will be concatenate
        # with each feature's lstm final hidden state to feed into the last mlp)
        self.dense1 = nn.Linear(self.static_feature_dim, self.static_embeddings_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.dropout_p)
        self.dense2 = nn.Linear(self.static_embeddings_dim, self.static_embeddings_dim)
        self.dropout2 = nn.Dropout(p=self.dropout_p)
        self.static_embeddings_list = torch.empty((1, self.batch_size, self.static_embeddings_dim)).cuda()

        # Below is the final MLP
        # The input for this MLP is the concatenation of each feature's lstm's final hidden state and
        # the static embeddings learned from the first MLP
        for f_id in range(self.feature_num):
            model_i = nn.Sequential(
                nn.Linear(self.static_embeddings_dim+self.hidden_dim_lstm, 2*(self.static_embeddings_dim+self.hidden_dim_lstm)),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(2*(self.static_embeddings_dim+self.hidden_dim_lstm), 21),  # 21 is the target size (20 for icd9 labels, 1 for y_mortality)
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p)
            )
            self.mlp_list.append(model_i.cuda())

    def forward(self, input_data):
        # To get static variables for all the features: input_data[0][0]
        # To get target data for all the features: input_data[0][2]
        # To get ith lstm's training data for feature i: input_data[i][1] (i starts from 0)

        features_embedding_list = []  # each element in this list trained from lstm has shape: [1,10,3]
        predict_list = torch.zeros((self.feature_num, self.batch_size, 21)).cuda()
        static_var_for_one_batch = input_data[0][0]
        target_for_all_features = input_data[0][2]
        static_embedding_list = torch.zeros((1, self.batch_size, self.static_embeddings_dim)).cuda()
        # get each temporal feature's embeddings (with batch_size == 10)
        for cur_lstm in range(self.feature_num):

            cur_output, (cur_final_hidden_state, cur_final_cell_state) = self.lstm_list[cur_lstm](input_data[cur_lstm][1], self.initial_hidden_cell_list[cur_lstm])
            features_embedding_list.append(cur_final_hidden_state)

        # get static variable's embeddings for a whole batch(batch_size == 10)
        for cur_epoch in range(self.batch_size):
            temp_result = self.dense1(static_var_for_one_batch[cur_epoch])
            temp_result = self.relu(temp_result)
            temp_result = self.dropout1(temp_result)
            temp_result = self.dense2(temp_result)
            temp_result = self.relu(temp_result)
            temp_result = self.dropout2(temp_result)
            static_embedding_list[0][cur_epoch] = temp_result.cuda()

        for f_id in range(self.feature_num):
            cur_input_for_final_mlps = torch.cat((features_embedding_list[f_id], static_embedding_list), 2)
            predict_list[f_id] = self.mlp_list[f_id](cur_input_for_final_mlps)

        # "predict_list" is a feature_num-length list, each element is a tensor with shape [1, batch_size, 21] representing the predict for each feature"
        return predict_list

    def Loss(self, target_list, predict_list):
        new_target_list = torch.zeros((self.feature_num, self.batch_size, 21)).cuda() # define "new_target_list" is aimed at unite the target size and the predict size
        for t_id in range(len(target_list)):
            for f_id in range(self.feature_num):
                new_target_list[f_id][t_id] = target_list[t_id]
        loss = nn.functional.binary_cross_entropy_with_logits(predict_list, new_target_list)
        # regularization term will be computed in TrainProcess
        return loss


def my_collate(batch, batch_size, seq_len):
    static_data_batch = [item[0] for item in batch]
    target_batch = [item[3] for item in batch]
    temporal_feature_batch = torch.zeros((batch_size, seq_len, 1))
    mask_feature_batch = torch.zeros((batch_size, seq_len, 1))
    for id, item in enumerate(batch):
        temporal_feature_batch[id] = item[1].view(seq_len, 1)
        mask_feature_batch[id] = item[2].view(seq_len, 1)
    input_lstm_batch = torch.cat((temporal_feature_batch, mask_feature_batch), 2)
    return static_data_batch, input_lstm_batch, target_batch


def Get_dataloaders(num_of_features, batch_size, seq_len):
    total_data_size = len(datasets_12_features[0])
    train_size = int(0.65 * total_data_size)
    test_size = int(0.25 * total_data_size)
    val_size = int(0.1 * total_data_size)
    left_size = total_data_size - (train_size + test_size + val_size)
    train_loader_list = []
    val_loader_list = []
    test_loader_list = []
    for i in range(num_of_features):
        train_set, test_set, val_set = random_split(datasets_12_features[i],
                                                    [train_size + left_size, test_size, val_size],
                                                    generator=torch.Generator().manual_seed(42))
        train_loader_list.append(
            DataLoader(train_set, batch_size=batch_size, shuffle=True,
                       collate_fn=partial(my_collate, batch_size=batch_size, seq_len=seq_len), drop_last=True))
        val_loader_list.append(
            DataLoader(val_set, batch_size=batch_size, shuffle=True,
                       collate_fn=partial(my_collate, batch_size=batch_size, seq_len=seq_len), drop_last=True))
        test_loader_list.append(
            DataLoader(test_set, batch_size=batch_size, shuffle=True,
                       collate_fn=partial(my_collate, batch_size=batch_size, seq_len=seq_len), drop_last=True))
    return train_loader_list, val_loader_list, test_loader_list


def TestProcess(test_loader_list, model, feature_num, targets_type, batch_size): # targets_type == 21
    perf_auprc = torch.zeros((feature_num, targets_type))  # perf_auprc[i][j] denotes feature i's target type j's auprc on test set
    perf_auroc = torch.zeros((feature_num, targets_type))  # perf_auroc[i][j] denotes feature i's target type j's auroc on test set
    testloader_length = len(test_loader_list[0])
    targets_for_auprc_auroc = torch.zeros((feature_num, targets_type, testloader_length*batch_size)).cuda()  # (13, 21, testset_size)
    predicts_for_auprc_auroc = torch.zeros((feature_num, targets_type, testloader_length*batch_size)).cuda() # (13, 21, testset_size)
    model.eval()
    with torch.no_grad():
        for t, data in enumerate(zip(test_loader_list[0], test_loader_list[1], test_loader_list[2], test_loader_list[3],
                                     test_loader_list[4], test_loader_list[5], test_loader_list[6], test_loader_list[7],
                                     test_loader_list[8], test_loader_list[9], test_loader_list[10], test_loader_list[11])):
            predict = model(data)  # (size: [12, batch_size, 21])
            cur_target = data[0][2]  # (size: batch_size-length size, each element is a (21,) tensor)
            new_target_list = torch.zeros((batch_size, targets_type)).cuda()  # (size: [10, 21])
            for b_id in range(batch_size):
                new_target_list[b_id] = cur_target[b_id]
            for f_id in range(feature_num):
                for t_id in range(targets_type):

                    targets_for_auprc_auroc[f_id][t_id][t*batch_size:(t+1)*batch_size] = new_target_list[:, t_id]
                    predicts_for_auprc_auroc[f_id][t_id][t*batch_size:(t+1)*batch_size] = predict[f_id, :, t_id]

    for f_id in range(feature_num):
        for t_id in range(targets_type):
            perf_auprc[f_id][t_id] = average_precision_score(targets_for_auprc_auroc[f_id][t_id].cpu(),
                                                             predicts_for_auprc_auroc[f_id][t_id].cpu())
            perf_auroc[f_id][t_id] = roc_auc_score(targets_for_auprc_auroc[f_id][t_id].cpu(),
                                                   predicts_for_auprc_auroc[f_id][t_id].cpu())
    print("each feature's auprc on each target (test set):\n")
    print(perf_auprc)
    print("each feature's auroc on each target (test set):\n")
    print(perf_auroc)


def main(search_space, num_of_features, static_feature_dim, hyperparam_tune_times, targets_type, seq_len):
    best_loss = 0
    best_config = 0
    final_test_loader_list = 0
    test_batch_size = 0
    for tune_time in range(hyperparam_tune_times):
        config = {k: random.sample(v, 1)[0] for k, v in search_space.items()}
        mix_model = MIMICIIIWithSepFeatures(num_of_features, static_feature_dim, config["static_embeddings_dim"],
                                        config["hidden_dim_lstm"], config["dropout_p"], config["batch_size"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        mix_model.to(device)
        optimizer = optim.Adam(mix_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        # Get dataloaders
        train_loader_list, val_loader_list, test_loader_list = Get_dataloaders(num_of_features, config["batch_size"], seq_len)
        print("Train Loader and Val Loader are made successfully\n")
        print("Begin model training:\n")  # training 10 times on each hyperparameter configuration
        for epoch in range(10):
            for t, data in enumerate(
                    zip(train_loader_list[0], train_loader_list[1], train_loader_list[2], train_loader_list[3],
                        train_loader_list[4], train_loader_list[5], train_loader_list[6], train_loader_list[7],
                        train_loader_list[8], train_loader_list[9], train_loader_list[10], train_loader_list[11])):
                # in each feature's data -- 0: static_batch, 1: input_lstm_batch, 2: target_batch

                predict = mix_model(data)
                cur_target = data[0][2]
                loss = mix_model.Loss(cur_target, predict)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (t + 1) % 3000 == 0:
                    print("current epoch: ", (t + 1) * config["batch_size"], " current training loss: ", loss.item())
        print("Begin model validation:\n")
        val_loss = 0.0
        val_steps = 0
        for val_id, val_data in enumerate(
                zip(val_loader_list[0], val_loader_list[1], val_loader_list[2], val_loader_list[3],
                    val_loader_list[4], val_loader_list[5], val_loader_list[6], val_loader_list[7],
                    val_loader_list[8], val_loader_list[9], val_loader_list[10], val_loader_list[11])):
            predict = mix_model(val_data)
            cur_target = val_data[0][2]
            loss = mix_model.Loss(cur_target, predict)
            val_loss += loss.item()
            val_steps += 1
        val_loss_avg = val_loss / val_steps
        if (best_loss == 0) or (val_loss_avg < best_loss):
            best_loss = val_loss_avg
            best_config = config
            torch.save(mix_model.state_dict(), './best_sep_feature_model_weights.pth')
            final_test_loader_list = test_loader_list
            test_batch_size = config["batch_size"]
            print("current config is the best! config is like:\n")
            print(best_config)
            print("current validation loss is:\n")
            print(best_loss)
        else:
            continue

    best_model = MIMICIIIWithSepFeatures(num_of_features, static_feature_dim, best_config["static_embeddings_dim"],
                                        best_config["hidden_dim_lstm"], best_config["dropout_p"], best_config["batch_size"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_model.to(device)
    best_model.load_state_dict(torch.load('./best_sep_feature_model_weights.pth'))
    TestProcess(final_test_loader_list, best_model, num_of_features, targets_type, test_batch_size)


num_of_features = 13
static_feature_dim = 5
targets_type = 21
hyperparam_tune_times = 20
seq_len = 24

search_space = {"static_embeddings_dim": list(range(2, 7)),
                "hidden_dim_lstm": list(range(2, 7)),
                "batch_size": [5, 10, 15, 20],
                "dropout_p": [0.1, 0.2, 0.3],
                "learning_rate": list(np.logspace(np.log10(0.001), np.log10(0.5), base=10, num=20)),
                "weight_decay": list(np.linspace(0.05, 0.5, 10))}

main(search_space, num_of_features, static_feature_dim, hyperparam_tune_times, targets_type, seq_len)
print("Finish!\n")


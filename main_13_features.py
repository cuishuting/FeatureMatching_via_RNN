import torch
import numpy as np
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split, DataLoader
from GetDataset_Each_Feature import MIMICIIIEachFeature
from sklearn.metrics import average_precision_score, roc_auc_score
import random
from torch.nn import BCEWithLogitsLoss
from torchsampler import ImbalancedDatasetSampler


data_dir = os.path.abspath("./MIMIC_timeseries/24hours/series/imputed-normed-ep_1_24.npz")
datasets_12_features = []
for f in range(12):
    datasets_12_features.append(MIMICIIIEachFeature(data_dir, f))


class TempCatStatic(nn.Module):
    def __init__(self, input_dim_lstm, hidden_dim_lstm, static_dim, target_dim, batch_size, static_embedding_dim, seq_len):
        super(TempCatStatic, self).__init__()
        self.input_dim_lstm = input_dim_lstm
        self.hidden_dim_lstm = hidden_dim_lstm
        self.static_dim = static_dim
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.static_embedding_dim = static_embedding_dim
        self.seq_len = seq_len
        # lstm part used to get temporal embeddings
        self.lstm = nn.LSTMCell(self.input_dim_lstm, hidden_dim_lstm) # input_dim_lstm == 2, one for temporal feature and the other for corresponding mask
        self.initial_hidden_cell = (torch.randn((self.batch_size, self.hidden_dim_lstm)).cuda(),
                                    torch.randn((self.batch_size, self.hidden_dim_lstm)).cuda())

        # mlp part used to get static features' embeddings
        self.dense1 = nn.Linear(self.static_dim, self.static_embedding_dim)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dense2 = nn.Linear(self.static_embedding_dim, self.static_embedding_dim)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dense3 = nn.Linear(self.static_embedding_dim, self.static_embedding_dim)
        self.dropout3 = nn.Dropout(p=0.25)

        # mlp part used to predict 21 targets after concatenating temporal embeddings and static embeddings
        self.LR_layer = nn.Linear(self.static_embedding_dim+self.hidden_dim_lstm, self.target_dim)

    def forward(self, input_data):
        static_feature = input_data[0]  # shape: [batch_size, 5]
        temporal_feature = input_data[1].view(-1, self.seq_len, 1)  # shape: [batch_size, 24] -> [batch_size, 24, 1]
        temporal_mask = input_data[2].view(-1, self.seq_len, 1)  # shape: [batch_size, 24] -> [batch_size, 24, 1]
        targets = input_data[3]  # shape: [batch_size, 21]

        # get temporal embeddings: h
        input_lstm = torch.cat((temporal_feature, temporal_mask), 2) # shape: [batch_size, 24, 2]
        h, c = self.initial_hidden_cell
        for t in range(self.seq_len):
            cur_input = input_lstm[:, t, :] # shape: [batch_size, 2]
            h, c = self.lstm(cur_input, (h, c))

        # get static embeddings: s
        s = self.dense1(static_feature)
        s = nn.ReLU()(s)
        s = self.dropout1(s)
        s = self.dense2(s)
        s = nn.ReLU()(s)
        s = self.dropout2(s)
        s = self.dense3(s)
        s = nn.ReLU()(s)
        s = self.dropout3(s)

        # concatenate temporal embeddings and static embeddings to predict final targets
        # h: [batch_size, hidden_dim_lstm]
        # s: [batch_size, static_embedding_dim]
        h_cat_s = torch.cat((h, s), 1)
        predicts = self.LR_layer(h_cat_s)

        # compute loss function
        criterion = BCEWithLogitsLoss()
        loss = criterion(predicts, targets.float())

        return {"predicts": predicts, "loss": loss}




def Get_dataloaders(num_of_features, batch_size, targets_dim):
    total_data_size = len(datasets_12_features[0])
    train_size = int(0.65 * total_data_size)
    test_size = int(0.25 * total_data_size)
    val_size = int(0.1 * total_data_size)
    left_size = total_data_size - (train_size + test_size + val_size)
    train_loader_list = []
    val_loader_list = []
    test_loader_list = []
    each_target_pos = np.zeros(targets_dim)
    for i in range(num_of_features):
        train_set, test_set, val_set = random_split(datasets_12_features[i],
                                                    [train_size + left_size, test_size, val_size],
                                                    generator=torch.Generator().manual_seed(42))
        if i == 0:
            for test_item in test_set:
                for t_id in range(targets_dim):
                    each_target_pos[t_id] += test_item[3][t_id]

        train_loader_list.append(
            DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True))
        val_loader_list.append(
            DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True))
        test_loader_list.append(
            DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True))
    baseline_auprc = each_target_pos / test_size
    return train_loader_list, val_loader_list, test_loader_list, baseline_auprc


def TestProcess(test_loader_list, model_list, feature_num, targets_type, batch_size): # targets_type == 21
    perf_auprc = torch.zeros((feature_num, targets_type))  # perf_auprc[i][j] denotes feature i's target type j's auprc on test set
    perf_auroc = torch.zeros((feature_num, targets_type))  # perf_auroc[i][j] denotes feature i's target type j's auroc on test set
    testloader_length = len(test_loader_list[0])
    targets_for_auprc_auroc = torch.zeros((feature_num, targets_type, testloader_length*batch_size)).cuda()  # (12, 21, testset_size)
    predicts_for_auprc_auroc = torch.zeros((feature_num, targets_type, testloader_length*batch_size)).cuda() # (12, 21, testset_size)
    with torch.no_grad():
        for t, data in enumerate(zip(test_loader_list[0], test_loader_list[1], test_loader_list[2], test_loader_list[3],
                                     test_loader_list[4], test_loader_list[5], test_loader_list[6], test_loader_list[7],
                                     test_loader_list[8], test_loader_list[9], test_loader_list[10], test_loader_list[11])):
            for f_id in range(feature_num):
                predict = model_list[f_id](data[f_id])["predicts"]  # [batch_size, 21]
                cur_target = data[f_id][3] # [batch_size, 21]
                for t_id in range(targets_type):
                    targets_for_auprc_auroc[f_id][t_id][t*batch_size:(t+1)*batch_size] = cur_target[:, t_id]
                    predicts_for_auprc_auroc[f_id][t_id][t*batch_size:(t+1)*batch_size] = predict[:, t_id]

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
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    for tune_time in range(hyperparam_tune_times):
        config = {k: random.sample(v, 1)[0] for k, v in search_space.items()}
        model_list = []
        optimizer_list = []

        for i in range(num_of_features):
            model_for_feature_i = TempCatStatic(2, config["hidden_dim_lstm"], static_feature_dim, targets_type, config["batch_size"], config["static_embeddings_dim"], seq_len)
            model_for_feature_i.to(device)
            model_list.append(model_for_feature_i)
            optimizer_list.append(optim.Adam(model_list[i].parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]))

        # Get dataloaders
        train_loader_list, val_loader_list, test_loader_list, baseline_auprc = Get_dataloaders(num_of_features, config["batch_size"], targets_type)
        print("current tune time is: ", (tune_time+1), " current baseline_auprc is: \n", baseline_auprc)
        print("Train Loader and Val Loader are made successfully\n")
        print("Begin model training:\n")  # training 10 times on each hyperparameter configuration

        for epoch in range(10):
            for t, data in enumerate(
                    zip(train_loader_list[0], train_loader_list[1], train_loader_list[2], train_loader_list[3],
                        train_loader_list[4], train_loader_list[5], train_loader_list[6], train_loader_list[7],
                        train_loader_list[8], train_loader_list[9], train_loader_list[10], train_loader_list[11])):
                # in each feature's data -- 0: static_batch, 1: input_lstm_batch, 2: target_batch

                for f_id in range(num_of_features):
                    model_list[f_id].train()
                    optimizer_list[f_id].zero_grad()
                    each_loss = model_list[f_id](data[f_id])["loss"]
                    each_loss.backward()
                    optimizer_list[f_id].step()
                    if (t + 1) % 3000 == 0:
                        print("current epoch: ", (t + 1) * config["batch_size"], " current feature id: ", (f_id + 1)," current training loss: ", each_loss.item())

        print("Begin model validation:\n")
        val_loss = 0.0
        val_steps = 0

        for val_id, val_data in enumerate(
                zip(val_loader_list[0], val_loader_list[1], val_loader_list[2], val_loader_list[3],
                    val_loader_list[4], val_loader_list[5], val_loader_list[6], val_loader_list[7],
                    val_loader_list[8], val_loader_list[9], val_loader_list[10], val_loader_list[11])):
            for f_id in range(num_of_features):
                model_list[f_id].eval()
                val_loss += model_list[f_id](val_data[f_id])["loss"].item()
                val_steps += 1

            # predict = mix_model(val_data)
            # cur_target = val_data[0][2]
            # loss = mix_model.Loss(cur_target, predict)
            # val_loss += loss.item()
            # val_steps += 1

        val_loss_avg = val_loss / num_of_features / val_steps
        if (best_loss == 0) or (val_loss_avg < best_loss):
            best_loss = val_loss_avg
            best_config = config
            for f_id in range(num_of_features):
                cur_weight_file_path = './' + 'f_' + str(f_id+1) + '_model_weights.pth'
                torch.save(model_list[f_id].state_dict(), cur_weight_file_path)
            final_test_loader_list = test_loader_list
            test_batch_size = config["batch_size"]
            print("current config is the best! config is like:\n")
            print(best_config)
            print("current validation loss is:\n")
            print(best_loss)
        else:
            continue

    best_model_list = []
    for i in range(num_of_features):
        best_model_for_feature_i = TempCatStatic(2, best_config["hidden_dim_lstm"], static_feature_dim, targets_type,
                                            best_config["batch_size"], best_config["static_embeddings_dim"], seq_len)
        best_model_for_feature_i.to(device)
        best_model_for_feature_i.load_state_dict(torch.load('./' + 'f_' + str(f_id+1) + '_model_weights.pth'))
        best_model_list.append(best_model_for_feature_i)

    TestProcess(final_test_loader_list, best_model_list, num_of_features, targets_type, best_config["batch_size"])


num_of_features = 12
static_feature_dim = 5
targets_type = 21
hyperparam_tune_times = 10
seq_len = 24

search_space = {"static_embeddings_dim": list(range(8, 11)),
                "hidden_dim_lstm": list(range(4, 11)),
                "batch_size": [5, 10, 15, 20],
                "learning_rate": list(np.logspace(np.log10(0.001), np.log10(0.5), base=10, num=20)),
                "weight_decay": list(np.linspace(0.05, 0.5, 10))}

main(search_space, num_of_features, static_feature_dim, hyperparam_tune_times, targets_type, seq_len)
print("Finish!\n")


import torch
import numpy as np
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split, DataLoader
from GetDataset_All import OrgMIMICIIIDataset
from sklearn.metrics import average_precision_score, roc_auc_score
import random
from torch.nn import BCEWithLogitsLoss


class Sep12LSTMsOneMLP(nn.Module):
    def __init__(self, batch_size, seq_len, temporal_feature_num, hidden_size_lstm, static_dim, static_embeddings_dim, hidden_dim_mlp, targets_dim):
        super(Sep12LSTMsOneMLP, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_temporal = temporal_feature_num
        self.hidden_size_lstm = hidden_size_lstm
        self.static_dim = static_dim
        self.static_embedding_dim = static_embeddings_dim
        self.targets_dim = targets_dim
        self.hidden_dim_mlp = hidden_dim_mlp
        # 12 lstms for 12 temporal features
        self.lstm_list = nn.ModuleList([nn.LSTMCell(2, self.hidden_size_lstm) for f in range(self.num_temporal)])
        # mlp for getting static embeddings
        self.static_layer1 = nn.Linear(self.static_dim, self.static_embedding_dim)
        self.static_drop1 = nn.Dropout(p=0.25)
        self.static_layer2 = nn.Linear(self.static_embedding_dim, self.static_embedding_dim)
        self.static_drop2 = nn.Dropout(p=0.25)
        self.static_layer3 = nn.Linear(self.static_embedding_dim, self.static_embedding_dim)
        self.static_drop3 = nn.Dropout(p=0.25)
        # mlp for final target predicting
        self.dense1 = nn.Linear(self.hidden_size_lstm*self.num_temporal + self.static_embedding_dim, self.hidden_dim_mlp)
        self.final_drop1 = nn.Dropout(p=0.25)
        self.dense2 = nn.Linear(self.hidden_dim_mlp, self.hidden_dim_mlp)
        self.final_drop2 = nn.Dropout(p=0.25)
        self.dense3 = nn.Linear(self.hidden_dim_mlp, self.targets_dim)
        self.final_drop3 = nn.Dropout(p=0.25)

    def forward(self, input_data):
        sample_static_features = input_data[0] # shape: [batch_size, 5]
        sample_temporal_features = input_data[1] # shape: [batch_size, 24, 12]
        sample_temporal_masks = input_data[2] # shape: [batch_size, 24, 12]
        sample_targets = input_data[3] # shape: [batch_size, 2]
        h_list = [] # hidden state list for 12 lstms
        c_list = [] # cell state list for 12 lstms

        # get final hidden state for each temporal feature: h_list[f] for temporal feature f
        for f, lstm in enumerate(self.lstm_list):
            h_list.append(torch.randn((self.batch_size, self.hidden_size_lstm)).cuda())
            c_list.append(torch.randn((self.batch_size, self.hidden_size_lstm)).cuda())
            cur_temporal_f = sample_temporal_features[:, :, f].reshape(self.batch_size, self.seq_len, 1)  # shape: [batch_size, 24, 1]
            cur_temporal_m = sample_temporal_masks[:, :, f].reshape(self.batch_size, self.seq_len, 1) # shape: [batch_size, 24, 1]
            cur_input_lstm = torch.cat((cur_temporal_f, cur_temporal_m), 2) # shape: [batch_size, 24, 2]
            for t in range(self.seq_len):
                cur_input = cur_input_lstm[:, t, :] # shape: [batch_size, 2(input_size)]
                h_list[f], c_list[f] = self.lstm_list[f](cur_input, (h_list[f], c_list[f])) # h_list[f]'s shape: [batch_size, hidden_size_lstm]
        # get static embeddings: s_emb
        s_emb = self.static_layer1(sample_static_features)
        s_emb = nn.ReLU()(s_emb)
        s_emb = self.static_drop1(s_emb)
        s_emb = self.static_layer2(s_emb)
        s_emb = nn.ReLU()(s_emb)
        s_emb = self.static_drop2(s_emb)
        s_emb = self.static_layer3(s_emb)
        s_emb = nn.ReLU()(s_emb)
        s_emb = self.static_drop3(s_emb) # shape: [batch_size, static_embedding_dim]

        # concat s_emb & final hidden state of each lstm to predict the final 2-dim target
        cat_temp_emb = torch.cat(h_list, 1) # shape: [batch_size, hidden_size_lstm*temporal_feature_num]
        temp_cat_s = torch.cat((cat_temp_emb, s_emb), 1)
        # shape: [batch_size, s_emb_dim + hidden_lstm_emb*temporal_feature_dim]
        predict = self.dense1(temp_cat_s)
        predict = nn.ReLU()(predict)
        predict = self.final_drop1(predict)
        predict = self.dense2(predict)
        predict = nn.ReLU()(predict)
        predict = self.final_drop2(predict)
        predict = self.dense3(predict)
        predict = nn.ReLU()(predict)
        predict = self.final_drop3(predict) # shape: [batch_size, 2]

        criterion = BCEWithLogitsLoss()
        loss = criterion(predict, sample_targets.float())

        return {"predicts": predict, "loss": loss}


def train(train_loader, model, optimizer, device, batch_size):
    model.train()
    for t, data in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)
        optimizer.zero_grad()
        cur_loss = model(data)["loss"]
        cur_loss.backward()
        optimizer.step()
        if (t + 1) % 3000 == 0:
            print("current data: ", (t + 1) * batch_size, " current training loss: ", cur_loss.item())


def val(val_loader, model, device):
    val_loss_sum = 0
    model.eval()
    with torch.no_grad():
        for v_id, v_data in enumerate(val_loader):
            for i in range(len(v_data)):
                v_data[i] = v_data[i].to(device)
            val_loss_sum += model(v_data)["loss"]
    avg_val_loss = val_loss_sum / len(val_loader)
    return avg_val_loss


def test(test_loader, model, device, targets_type, batch_size):
    targets_for_auprc_auroc = torch.zeros((targets_type, len(test_loader) * batch_size)).cuda()
    predicts_for_auprc_auroc = torch.zeros((targets_type, len(test_loader) * batch_size)).cuda()
    each_target_pos = np.zeros(targets_type)
    perf_auprc = torch.zeros(targets_type)
    perf_auroc = torch.zeros(targets_type)
    with torch.no_grad():
        for t_id, test_data in enumerate(test_loader):
            for i in range(len(test_data)):
                test_data[i] = test_data[i].to(device)
            cur_predicts = model(test_data)["predicts"]  # shape: [batch_size, 2]
            cur_targets = test_data[3] # shape: [batch_size, 2]
            pos_num_each_target = (cur_targets == 1).sum(dim=0) # 1-dim tensor with length "targets_type"
            for target_id in range(targets_type):
                each_target_pos[target_id] += pos_num_each_target[target_id].item()
                predicts_for_auprc_auroc[target_id][t_id*batch_size : (t_id+1)*batch_size] = cur_predicts[:, target_id]
                targets_for_auprc_auroc[target_id][t_id*batch_size : (t_id+1)*batch_size] = cur_targets[:, target_id]

    for target_id in range(targets_type):
        perf_auprc[target_id] = average_precision_score(targets_for_auprc_auroc[target_id].cpu(), predicts_for_auprc_auroc[target_id].cpu())
        perf_auroc[target_id] = roc_auc_score(targets_for_auprc_auroc[target_id].cpu(), predicts_for_auprc_auroc[target_id].cpu())

    print("each target's baseline:\n")
    print(each_target_pos / (targets_for_auprc_auroc.shape[1]))
    print("targets' auprc:\n")
    print(perf_auprc)
    print("targets' auroc:\n")
    print(perf_auroc)


def main(data_addr, search_space, tune_times, epochs, seq_len, temporal_feature_num, static_feature_num, targets_dim):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using {device} device")
    best_val_loss = 0
    best_config = 0
    data_dir = os.path.abspath(data_addr)
    dataset_all_features = OrgMIMICIIIDataset(data_dir)
    total_data_size = len(dataset_all_features)
    test_size = int(0.25 * total_data_size)
    val_size = int(0.1 * total_data_size)
    train_size = total_data_size - (test_size + val_size)
    train_set, test_set, val_set = random_split(dataset_all_features, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))
    final_test_loader = 0
    """
    below is the test on AUG 21st to see whether increasing the hidden state size for the final mlp will improve the test accuracy
    """

    """
    the end of the test on AUG 21st
    """
    for tune_time in range(tune_times):
        config = {k: random.sample(v, 1)[0] for k, v in search_space.items()}
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        model = Sep12LSTMsOneMLP(config["batch_size"], seq_len, temporal_feature_num, config["hidden_dim_lstm"], static_feature_num, config["static_embeddings_dim"], config["hidden_dim_mlp"], targets_dim)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        for epoch in range(epochs):
            train(train_loader, model, optimizer, device, config["batch_size"])
        cur_avg_val_loss = val(val_loader, model, device)
        if (best_val_loss == 0) or (cur_avg_val_loss < best_val_loss):
            best_val_loss = cur_avg_val_loss
            best_config = config
            model_weight_file_path = "./best_model_weights_08_17.pth"
            torch.save(model.state_dict(), model_weight_file_path)
            final_test_loader = test_loader
            print("current config is the best! config is like:\n")
            print(best_config)
            print("current best validation loss is:\n")
            print(best_val_loss)

    best_model = Sep12LSTMsOneMLP(best_config["batch_size"], seq_len, temporal_feature_num,
                                  best_config["hidden_dim_lstm"], static_feature_num,
                                  best_config["static_embeddings_dim"], best_config["hidden_dim_mlp"], targets_dim).to(device)
    best_model.load_state_dict(torch.load("./best_model_weights_08_17.pth"))
    test(final_test_loader, best_model, device, targets_dim, best_config["batch_size"])


search_space = {"static_embeddings_dim": list(range(3, 11)),
                "hidden_dim_lstm": list(range(2, 11)),
                "hidden_dim_mlp": list(range(4, 21)), # final mlp's(used to predict final target) hidden dim
                "batch_size": [5, 10, 15, 20],
                "learning_rate": list(np.logspace(np.log10(0.001), np.log10(0.5), base=10, num=20)),
                "weight_decay": list(np.linspace(0.05, 0.5, 10))}
data_addr = "./MIMIC_timeseries/24hours/series/imputed-normed-ep_1_24.npz"
tune_times = 15
epochs = 10
seq_len = 24
temporal_feature_num = 12
static_feature_num = 5
targets_dim = 2
main(data_addr, search_space, tune_times, epochs, seq_len, temporal_feature_num, static_feature_num, targets_dim)




import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
from GetDataset import dataset
import matplotlib.pyplot as plt







class CrossNet_MIMIC_III(nn.Module):
    def __init__(self, feature_dim, hidden_dim_lstm, hidden_size_mlp):
        super(CrossNet_MIMIC_III, self).__init__()
        # initialize for the lstm model
        self.lstm = nn.LSTM(feature_dim, hidden_dim_lstm)
        self.initial_hidden_cell = (torch.randn(1, 1, hidden_dim_lstm).cuda(), torch.randn(1, 1, hidden_dim_lstm).cuda())

        # initialize for the MLP after the lstm model
        self.input_size_mlp = hidden_dim_lstm
        self.hidden_size_mlp = hidden_size_mlp
        self.fc1 = nn.Linear(self.input_size_mlp, self.hidden_size_mlp)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size_mlp, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        # param "input_data": time series data for one admission with shape(L,N,12),
        # L is the # of time stamps for this admission; N=1 in this case
        output_for_each_t, (final_hidden_state, final_cell_state) = self.lstm(input_data, self.initial_hidden_cell)
        # param "final_hidden_state" has shape (2, 1, hidden_dim)
        # 2 is because using "bidirectional LSTM"
        hidden = self.fc1(final_hidden_state)
        hidden = self.relu1(hidden)
        output = self.fc2(hidden)
        final_output = self.sigmoid(output)
        return final_output

    def Loss(self, target_list, predict_list, lambda1=0.01):
        loss = nn.functional.binary_cross_entropy_with_logits(predict_list, target_list.float())
        reg_term1 = lambda1 * torch.sum(torch.norm(self.fc1.weight.data, p=2))
        reg_term2 = lambda1 * torch.sum(torch.norm(self.fc2.weight.data, p=2))
        final_loss = loss + reg_term1 + reg_term2
        return final_loss


def TrainProcess(train_loader, model, optimizer, train_loss_list):
    total_loss = torch.tensor(0, dtype=torch.float).cuda()
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X[0][X[0] != X[0]] = 0
        output = model(X[0])
        target = y[0]
        loss = model.Loss(target, output[0,0])
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 500 == 0:
        #     loss_val = loss.item()
        #     print("training loss: ", loss_val, "current pos: ", batch)
    batch_size = len(train_loader)
    train_loss_list.append(total_loss.item() / batch_size)
    print(f"average training loss: {total_loss / batch_size}")

def TestProcess(test_loader, model, test_loss_list, test_accuracy_list):
    # Model Testing
    test_loss = torch.tensor(0, dtype=torch.float).cuda()
    correct_pred = 0
    test_batch_size = len(test_loader)
    model.eval()
    with torch.no_grad():
        for X,y in test_loader:
            X[0][X[0] != X[0]] = 0
            pred = model(X[0])
            true_y = y[0]
            test_loss += model.Loss(true_y, pred[0,0])
            if (pred.item() > 0.5 and true_y.item() == 1) or (pred.item() < 0.5 and true_y.item() == 0):
                correct_pred += 1
    avg_test_loss = test_loss / test_batch_size
    test_loss_list.append(avg_test_loss.item())
    accuracy = correct_pred / test_batch_size
    test_accuracy_list.append(accuracy)
    print(f"average test loss is: {avg_test_loss}\n")
    print(f"accuracy: {accuracy}\n")





total_data_size = len(dataset)
train_size = int(0.65 * total_data_size)
test_size = int(0.25 * total_data_size)
val_size = int(0.1 * total_data_size)
left_size = total_data_size - (train_size + test_size + val_size)
train_set, test_set, val_set = random_split(dataset, [train_size+left_size, test_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

# two classes's distribution in three loaders
y_dis_train = pd.DataFrame([y[0].item() for X,y in train_loader]).value_counts()
print("0/1 classes' distribution in train loader:\n")
print(y_dis_train)
y_dis_test = pd.DataFrame([y[0].item() for X,y in test_loader]).value_counts()
print("0/1 classes' distribution in test loader:\n")
print(y_dis_test)
y_dis_val = pd.DataFrame([y[0].item() for X,y in val_loader]).value_counts()
print("0/1 classes' distribution in val loader:\n")
print(y_dis_val)

# Model Training
feature_dim = 12
hidden_dim_lstm = 6
hidden_size_mlp = 4
learning_rate = 0.001
model = CrossNet_MIMIC_III(feature_dim, hidden_dim_lstm, hidden_size_mlp)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
train_loss_list = []
test_loss_list = []
test_accuracy_list = []


epochs = 500
for epoch in range(epochs):
    print(f"Epoch: {epoch+1}\n-------------------------------")
    TrainProcess(train_loader, model, optimizer, train_loss_list)
    TestProcess(test_loader, model, test_loss_list, test_accuracy_list)
print("Finish!\n")


torch.save(model.state_dict(), 'model_weights.pth')
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train loss")
plt.plot(np.arange(len(test_loss_list)), test_loss_list, label="test loss")
plt.plot(np.arange(len(test_accuracy_list)), test_accuracy_list, label="test accuracy")
plt.legend()
plt.xlabel("epochs")
plt.title("loss and accuracy")
plt.show()















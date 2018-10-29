import math

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

from lstm_model.utility import to_supervised, SeyfriedParser, MyConfig, BIWIParser, ConstVelModel
from lstm_model.kalman import MyKalman

np.random.seed(1)
config = MyConfig(n_past=8, n_next=8)
n_past = config.n_past
n_next = config.n_next
test_interval = 10
n_inp_features = 4  # x, y, vx, vy
n_out_features = 2

train_rate = 0.8
learning_rate = 4e-3
weight_decay = 4e-3
def_batch_size = 256
n_epochs = 1000

if torch.cuda.is_available():
    print("CUDA is available!")


class PredictorLSTM(nn.Module):
    def __init__(self, feature_size, pred_length, hidden_size, num_layers=1):
        super(PredictorLSTM, self).__init__()
        self.feature_size = feature_size
        self.pred_length = pred_length
        self.hidden_size = hidden_size
        self.n_layers = num_layers

        # Initialize Layers
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers).cuda()
        self.fc_out = nn.Linear(hidden_size, pred_length * 2).cuda()

        # it gives out just one location
        self.one_out = nn.Linear(hidden_size, 2).cuda()

        #self.lstm.weight_hh_l0.data.fill_(0)
        nn.init.xavier_uniform_(self.fc_out.weight)

        self.linear = nn.Linear(hidden_size, hidden_size).cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        self.relu = nn.ReLU().cuda()

        self.hidden = self.init_state()

        self.loss_func = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def init_state(self, minibatch_size=1):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.n_layers, minibatch_size, self.hidden_size).cuda(),
                torch.zeros(self.n_layers, minibatch_size, self.hidden_size).cuda())

    ### sequence-prediction
    # def forward(self, x):
    #     self.hidden = self.init_state()
    #     for xi in x:
    #         xi = xi.view(1, 1, self.feature_size)
    #         y, self.hidden = self.lstm(xi, self.hidden)
    #     # y = self.linear(y)
    #     # y = self.sigmoid(y)
    #     # y = self.relu(y)
    #     y = self.fc_out(y)
    #     return y

    ### goal-prediction
    def forward(self, x):
        self.hidden = self.init_state()
        for xi in x:
            xi = xi.view(1, 1, self.feature_size)
            y, self.hidden = self.lstm(xi, self.hidden)
        y = self.linear(y)
        # y = self.sigmoid(y)
        # y = self.relu(y)
        y = self.one_out(y)
        return y


def update(model, y_list, y_hat_list):
    y_stack = torch.stack(y_list, 2)
    y_hat_stack = torch.stack(y_hat_list, 2)
    loss = model.loss_func(y_hat_stack, y_stack)
    model.zero_grad()
    loss.backward()
    model.optimizer.step()
    return loss.item()


def train(model, ped_data, batch_size=def_batch_size):
    running_loss = 0
    running_cntr = 0
    y_list = []
    y_hat_list = []
    for ped_i in ped_data:
        ped_i_tensor = torch.FloatTensor(ped_i).cuda()
        seq_len = ped_i_tensor.size(0)
        for t in range(n_past, seq_len - n_next + 1):
            model.hidden = model.init_state()

            x = ped_i_tensor[t-n_past:t, 0:n_inp_features]

            # sequence-prediction
            # y = (ped_i_tensor[t:t + n_next, 0:2] - x[-1, 0:2]).view(n_next, 2)
            # y_hat = model(x).view(n_next, 2)

            # goal-prediction
            y = (ped_i_tensor[t + n_next-1, 0:2] - x[-1, 0:2]).view(1, 2)
            y_hat = model(x).view(1, 2)


            y_list.append(y)
            y_hat_list.append(y_hat)
            if len(y_list) >= batch_size:
                running_loss += update(model, y_list, y_hat_list) * batch_size
                running_cntr += batch_size
                y_list = []
                y_hat_list = []

    if len(y_list) > 0:
        running_loss += update(model, y_list, y_hat_list) * len(y_list)
        running_cntr += len(y_list)

    return running_loss / running_cntr


def test(model, ped_data):
    running_loss = 0
    running_cntr = 0
    cv_model = ConstVelModel()
    with torch.no_grad():
        for ii in range(len(ped_data)):
            ped_i_tensor = torch.FloatTensor(ped_data[ii]).cuda()
            for t in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
                x = ped_i_tensor[t-n_past:t, 0:n_inp_features]

                # y = (ped_i_tensor[t:t+n_next, 0:2] - x[-1, 0:2]).view(n_next, 2)
                # y_hat = model(x).view(n_next, 2)

                # Goal Prediction
                y = (ped_i_tensor[t+n_next-1, 0:2] - x[-1, 0:2]).view(1, 2)
                y_hat = model(x).view(1, 2)

                loss = model.loss_func(y_hat, y)
                running_loss += loss.item()
                running_cntr += 1

        # Display Results
        for ii in range(5, len(ped_data), 15):
            ped_i_tensor = torch.FloatTensor(ped_data[ii]).cuda()
            for t in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
                x = ped_i_tensor[t - n_past:t, 0:n_inp_features]
                x_np = x.cpu().data.numpy().reshape((n_past, n_inp_features))[:, 0:2]

                # y = (ped_i_tensor[t:t + n_next, 0:2] - x[-1, 0:2]).view(n_next, 2)
                # y_hat = model(x).view(n_next, 2)
                # y_np = y.cpu().data.numpy().reshape((n_next, 2))
                # y_hat_np = np.vstack((np.array([0, 0]), y_hat.cpu().data.numpy().reshape((n_next, 2)))) + x_np[-1, 0:2]

                # Goal Prediction
                y = (ped_i_tensor[t:t + n_next, 0:2] - x[-1, 0:2]).view(n_next, 2)
                y_hat = model(x).view(1, 2)
                y_np = y.cpu().data.numpy().reshape((n_next, 2))
                y_hat_np = np.vstack((np.array([0, 0]), y_hat.cpu().data.numpy().reshape((1, 2)))) + x_np[-1, 0:2]

                y_cv = np.vstack((x_np[-1, 0:2], cv_model.predict(x_np)))

                plt.plot(x_np[:, 0], x_np[:, 1], 'y--')
                plt.plot(y_cv[:, 0], y_cv[:, 1], 'r--')
                plt.plot(y_np[:, 0] + x_np[-1, 0], y_np[:, 1] + x_np[-1, 1], 'g--')
                plt.plot(y_hat_np[:, 0], y_hat_np[:, 1], 'b')
                plt.plot(x_np[-1, 0], x_np[-1, 1], 'mo', markersize=7, label='Start Point')

            plt.show()


    avg_loss = running_loss/running_cntr
    return avg_loss


def run():
    # parser = SeyfriedParser()
    # pos_data, vel_data, time_data = parser.load('/home/jamirian/workspace/crowd_sim/tests/sey01/sey01.sey')
    parser = BIWIParser()
    pos_data, vel_data, time_data = parser.load('/home/jamirian/workspace/crowd_sim/tests/eth/eth.wap')
    scale = parser.scale

    n_ped = len(pos_data)
    train_size = int(n_ped * train_rate)
    test_size = n_ped - train_size

    print('Dont forget to smooth the trajectories?')

    print('Yes! Smoothing the trajectories in train_set ...')
    for i in range(train_size):
        kf = MyKalman(1 / parser.new_fps, n_iter=5)
        pos_data[i], vel_data[i] = kf.smooth(pos_data[i])

    # Scaling
    data_set = list()
    for i in range(len(pos_data)):
        pos_data[i] = scale.normalize(pos_data[i], shift=True)
        vel_data[i] = scale.normalize(vel_data[i], shift=False)
        _pv_i = np.hstack((pos_data[i], vel_data[i]))
        data_set.append(_pv_i)
    train_set = np.array(data_set[:train_size])
    test_set = np.array(data_set[train_size:])

    model = PredictorLSTM(feature_size=n_inp_features, pred_length=n_next, hidden_size=128, num_layers=1)

    print("Train the model ...")
    for epoch in range(1, n_epochs+1):
        train_loss = train(model, train_set)
        train_loss = math.sqrt(train_loss) / (scale.sx)
        print('******* Epoch: [%d/%d], Loss: %.9f **********' % (epoch, n_epochs, train_loss))
        if epoch % test_interval == 0:
            test_loss = test(model, test_set)
            test_loss = math.sqrt(test_loss) / (scale.sx)
            print('====TEST====> MSE Loss:%.9f' % test_loss)


if __name__ == '__main__':
    run()

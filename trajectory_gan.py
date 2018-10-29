import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lstm_model.learning_utils import adjust_learning_rate
from lstm_model.utility import to_supervised, SeyfriedParser, MyConfig, BIWIParser, ConstVelModel
from lstm_model.kalman import MyKalman
import matplotlib.pyplot as plt

np.random.seed(1)
config = MyConfig(n_past=8, n_next=8)
n_past = config.n_past
n_next = config.n_next
n_inp_features = 4  # (x, y),  /vx, vy
n_out_features = 2

train_rate = 0.8
learning_rate = 4e-3
weight_decay = 4e-3
lambda_l2_loss = 5
lambda_dc_loss = 1
test_interval = 10  # FIXME << ===+   << ===+   << ===+
optim_betas = (0.9, 0.999)
def_batch_size = 128
n_epochs = 2000
noise_vec_len = 64


class Generator(nn.Module):
    def __init__(self, inp_len, out_len, noise_len=noise_vec_len):
        super(Generator, self).__init__()
        self.out_len = out_len

        self.use_noise = True      # For using it as a solely predictor
        self.noise_len = noise_len

        self.n_lstm_layers = 1
        self.inp_size_lstm = n_inp_features
        self.hidden_size_lstm = 48
        self.hidden_size_2 = 48
        self.is_blstm = False

        self.fc_in = nn.Sequential(nn.Linear(2, self.inp_size_lstm), nn.LeakyReLU(0.5)).cuda()

        # self.embedding = nn.Linear(inp_len * 2, hidden_size_1)
        self.lstm = nn.LSTM(input_size=self.inp_size_lstm, hidden_size=self.hidden_size_lstm,
                            num_layers=self.n_lstm_layers, batch_first=True, bidirectional=self.is_blstm).cuda()

        self.fc_out = nn.Linear(self.hidden_size_2, self.out_len * 2).cuda()

        # Hidden Layers
        self.fc_1 = nn.Sequential(
            nn.Linear(self.hidden_size_lstm * (1 + self.is_blstm) + self.noise_len * self.use_noise, self.hidden_size_2)
                                  , nn.Sigmoid()).cuda()

    def forward(self, obsv, noise):
        # input: (B, seq_len, 2)
        # noise: (B, N)
        batch_size = obsv.size(0)
        obsv = obsv[:, :, 0:self.inp_size_lstm]

        # ===== To use Dense Layer in the input ======
        # rr = []
        # for tt in range(obsv.size(1)):
        #     rr.append(self.fc_in(obsv[:, tt, :]))
        # obsv = torch.stack(rr, 1)

        # initialize hidden state: (num_layers, minibatch_size, hidden_dim)
        init_state = (torch.zeros(self.n_lstm_layers * (1 + self.is_blstm), batch_size, self.hidden_size_lstm).cuda(),
                      torch.zeros(self.n_lstm_layers * (1 + self.is_blstm), batch_size, self.hidden_size_lstm).cuda())

        (lstm_out, _) = self.lstm(obsv, init_state)  # encode the input
        last_lstm_out = lstm_out[:, -1, :].view(batch_size, 1, -1)  # just keep the last output: (batch_size, 1, H)

        # combine data with noise
        if self.use_noise:
            lstm_out_and_noise = torch.cat([last_lstm_out, noise.cuda().view(batch_size, 1, -1)], dim=2)
        else:
            lstm_out_and_noise = last_lstm_out

        u = self.fc_1(lstm_out_and_noise)

        # decode the data to generate fake sample
        pred_batch = self.fc_out(u).view(batch_size, self.out_len, 2)
        return pred_batch


class Discriminator(nn.Module):
    def __init__(self, obsv_len, pred_len):
        super(Discriminator, self).__init__()
        self.out_size_lstm = 32
        self.hidden_size_fc = 32
        self.obsv_len = obsv_len
        self.pred_len = pred_len
        self.n_lstm_layers = 1
        self.is_blstm_obsv = False
        self.is_blstm_pred = False
        self.inp_size_lstm_obsv = n_inp_features
        self.inp_size_lstm_pred = 2
        self.lstm_obsv = nn.LSTM(input_size=self.inp_size_lstm_obsv, hidden_size=self.out_size_lstm,
                                 num_layers=self.n_lstm_layers, batch_first=True,
                                 bidirectional=self.is_blstm_obsv).cuda()

        self.lstm_pred = nn.LSTM(input_size=self.inp_size_lstm_pred, hidden_size=self.out_size_lstm,
                                 num_layers=self.n_lstm_layers, batch_first=True,
                                 bidirectional=self.is_blstm_pred).cuda()

        self.fc_1 = nn.Sequential(nn.Linear(self.out_size_lstm * (1 + self.is_blstm_pred) +
                                            self.out_size_lstm * (1 + self.is_blstm_obsv), self.hidden_size_fc)
                                  , nn.LeakyReLU(0.5)).cuda()
        self.classifier = nn.Linear(self.hidden_size_fc, 1).cuda()

    def forward(self, obsv, pred):
        # obsv: (B, in_seq_len, F)
        # pred: (B, out_seq_len, F)
        batch_size = obsv.size(0)
        obsv = obsv[:, :, :self.inp_size_lstm_obsv]
        pred = pred[:, :, :self.inp_size_lstm_pred]


        # initialize hidden state of obsv_lstm: (num_layers, minibatch_size, hidden_dim)
        init_state1 = (torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_obsv), batch_size, self.out_size_lstm).cuda(),
                       torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_obsv), batch_size, self.out_size_lstm).cuda())

        # ! lstm_out: (batch_size, seq_len, H)
        (obsv_lstm_out, _) = self.lstm_obsv(obsv, init_state1)
        obsv_lstm_out = obsv_lstm_out[:, -1, :].view(batch_size, 1, -1)  # I just need the last output: (batch_size, 1, H)

        # initialize hidden state of pred_lstm: (num_layers, minibatch_size, hidden_dim)
        init_state2 = (torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_pred), batch_size, self.out_size_lstm).cuda(),
                       torch.zeros(self.n_lstm_layers * (1 + self.is_blstm_pred), batch_size, self.out_size_lstm).cuda())

        # ! lstm_out: (batch_size, seq_len, H)
        (pred_lstm_out, _) = self.lstm_pred(pred, init_state2)
        pred_lstm_out = pred_lstm_out[:, -1, :].view(batch_size, 1, -1)  # I just need the last output: (batch_size, 1, H)

        concat_lstm_outputs = torch.cat([obsv_lstm_out, pred_lstm_out], dim=2)

        u = self.fc_1(concat_lstm_outputs)

        # c: (batch_size, 1, 1)
        c = self.classifier(u)
        return c


# parser = SeyfriedParser()
# pos_data, vel_data, time_data = parser.load('/home/jamirian/workspace/crowd_sim/tests/sey01/sey01.sey')
parser = BIWIParser()
pos_data, vel_data, time_data = parser.load('/home/jamirian/workspace/crowd_sim/tests/eth/eth.wap')
# pos_data, vel_data, time_data = parser.load('/home/jamirian/workspace/crowd_sim/tests/hotel/hotel.wap')
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

dataset_x = []
dataset_y = []
for ped_i in train_set:
    ped_i_tensor = torch.FloatTensor(ped_i)  # .cuda()
    seq_len = ped_i_tensor.size(0)
    for t in range(n_past, seq_len - n_next + 1, 1):
        _x = ped_i_tensor[t - n_past:t, :]
        _y = (ped_i_tensor[t:t + n_next, 0:2] - _x[-1, 0:2])
        dataset_x.append(_x)
        dataset_y.append(_y)
dataset_x_tensor = torch.stack(dataset_x, 0)
dataset_y_tensor = torch.stack(dataset_y, 0)


def bce_loss(input_, target_):
    neg_abs = -input_.abs()
    _loss = input_.clamp(min=0) - input_ * target_ + (1 + neg_abs.exp()).log()
    return _loss.mean()


train = torch.utils.data.TensorDataset(dataset_x_tensor, dataset_y_tensor)
train_loader = DataLoader(train, batch_size=def_batch_size, shuffle=False, num_workers=4)

generator = Generator(n_past, n_next, noise_len=noise_vec_len)
discriminator = Discriminator(n_past, n_next)
discriminationLoss = nn.BCEWithLogitsLoss()  # Binary cross entropy
groundTruthLoss = nn.MSELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=optim_betas)


print("Train the model ...")
for epoch in range(1, n_epochs + 1):

    adjust_learning_rate(d_optimizer, epoch)
    adjust_learning_rate(g_optimizer, epoch)

    gen_loss_acc = 0
    dcr_loss_real = 0
    dcr_loss_fake = 0

    for i, (datas_x, datas_y) in enumerate(train_loader):
        xs = datas_x.cuda()
        ys = datas_y.cuda()

        batch_size = xs.size(0)
        noise = torch.rand(batch_size, noise_vec_len)

        # =============== Train Discriminator ================= #
        discriminator.zero_grad()

        y_hat_1 = generator(xs, noise)  # for updating discriminator
        y_hat_1.detach()
        c_hat_fake_1 = discriminator(xs, y_hat_1)  # classify fake samples
        # disc_loss_fakes = discriminationLoss(c_hat_fake_1, Variable(torch.zeros(batch_size, 1, 1).cuda()))
        disc_loss_fakes = bce_loss(c_hat_fake_1, (torch.zeros(batch_size, 1, 1) + random.uniform(0, 0.3)).cuda())
        disc_loss_fakes.backward()

        for _ in range(1 + 1 * (epoch < 50)):
            c_hat_real = discriminator(xs, ys)  # classify real samples
            # disc_loss_reals = discriminationLoss(c_hat_real, torch.ones(batch_size, 1, 1).cuda())
            disc_loss_reals = bce_loss(c_hat_real, (torch.ones(batch_size, 1, 1) * random.uniform(0.7, 1.2)).cuda())
            disc_loss_reals.backward()

        d_optimizer.step()

        dcr_loss_fake += disc_loss_fakes.item()
        dcr_loss_real += disc_loss_reals.item()

        # =============== Train Generator ================= #
        generator.zero_grad()
        discriminator.zero_grad()
        y_hat_2 = generator(xs, noise)  # for updating generator
        c_hat_fake_2 = discriminator(xs, y_hat_2)  # classify fake samples

        # gen_loss_fooling = discriminationLoss(c_hat_fake_2, Variable(torch.ones(batch_size, 1, 1).cuda()))
        gen_loss_fooling = bce_loss(c_hat_fake_2, (torch.ones(batch_size, 1, 1) * random.uniform(0.7, 1.2)).cuda())
        gen_loss_gt = groundTruthLoss(y_hat_2, ys) / n_next
        # print('L2 loss = ', gen_loss_gt.item())
        gen_loss = (gen_loss_gt * lambda_l2_loss) + (gen_loss_fooling * generator.use_noise * lambda_dc_loss)

        gen_loss.backward()
        g_optimizer.step()

        gen_loss_acc += gen_loss.item()

    gen_loss_acc /= i
    dcr_loss_fake /= i
    dcr_loss_real /= i

    print('epoch [%3d/%d], Generator Loss: %.6f , Gen Error: %.6f || Dis Loss: Fake= %5f, Real= %.5f'
          % (epoch, n_epochs, gen_loss_fooling.item(), gen_loss_gt * lambda_l2_loss, dcr_loss_fake, dcr_loss_real))

    # ====================== T E S T =======================
    if epoch % test_interval == 0:
        running_loss = 0
        running_cntr = 0
        cv_model = ConstVelModel()

        with torch.no_grad():
            for ii, ped_i in enumerate(test_set):
                ped_i_tensor = torch.FloatTensor(test_set[ii]).cuda()
                for t in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
                    x = ped_i_tensor[t - n_past:t, 0:n_inp_features].view(1, n_past, -1)

                    y = (ped_i_tensor[t:t+n_next, 0:2] - x[0, -1, 0:2]).view(n_next, 2)
                    y_hat = generator(x, torch.rand(1, noise_vec_len)).view(n_next, 2)

                    loss = groundTruthLoss(y_hat, y)
                    running_loss += loss.item()
                    running_cntr += 1

            # Display Results
            i0 = random.randint(2, 20)
            print("Don't forget to correct here and test the test set!")
            for ii in range(i0, len(train_set), 45):
                ped_i_tensor = torch.FloatTensor(train_set[ii]).cuda()
                for t in range(n_past, ped_i_tensor.size(0) - n_next + 1, n_past):
                    x = ped_i_tensor[t - n_past:t, 0:n_inp_features].view(1, n_past, -1)
                    x_np = x.cpu().data.numpy().reshape((n_past, n_inp_features))

                    y = (ped_i_tensor[t:t + n_next, 0:2] - x[0, -1, 0:2]).view(1, n_next, 2)
                    c_real = discriminator(x, y)
                    y_np = y.cpu().data.numpy().reshape((n_next, 2))

                    plt.plot(x_np[-1, 0], x_np[-1, 1], 'mo', markersize=7, label='Start Point')
                    if c_real > 0.5:
                        plt.plot(y_np[:, 0] + x_np[-1, 0], y_np[:, 1] + x_np[-1, 1], 'g--')
                    else:
                        plt.plot(y_np[:, 0] + x_np[-1, 0], y_np[:, 1] + x_np[-1, 1], 'y--')
                    # plt.plot(x_np[:, 0], x_np[:, 1], 'g--')

                    # =========== Const-Vel Prediction ========
                    y_cv = np.vstack((x_np[-1, 0:2], cv_model.predict(x_np[:, 0:2])))
                    plt.plot(y_cv[:, 0], y_cv[:, 1], 'c--')

                    # ============ Our Prediction =============
                    for kk in range(10):
                        y_hat = generator(x, torch.rand(1, noise_vec_len)).view(1, n_next, 2)
                        c_fake = discriminator(x, y_hat)
                        print(c_fake)
                        y_hat_np = np.vstack((np.array([0, 0]), y_hat.cpu().data.numpy().reshape((n_next, 2)))) + x_np[-1, 0:2]
                        if c_fake > 0.5:
                            plt.plot(y_hat_np[:, 0], y_hat_np[:, 1], 'b')
                        else:
                            plt.plot(y_hat_np[:, 0], y_hat_np[:, 1], 'r')

                    plt.ylim((0, 1))
                    plt.xlim((0, 1))

                plt.show()

        test_loss = running_loss / running_cntr
        print("Test loss = ", np.math.sqrt(test_loss) / scale.sx)

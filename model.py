import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, in_d, out_d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_d, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, out_d)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Learner(object):

    def __init__(self, in_d, out_d, lr):
        self.model = Net(in_d, out_d)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, x, t):
        x = self.model(x)
        loss = F.mse_loss(x, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        x = Variable(torch.Tensor(x))
        x = self.model(x)
        return x

""" this file contains all different models """
import torch
from torch import nn

# device = ("cuda" if torch.cuda.is_available() else "cpu")
from models.layers import vp_layer_coeffs, ada_hermite

device = None


class VPNet4(nn.Module):
    def __init__(self, penalty=10.0):
        super().__init__()
        self.linear1 = nn.Linear(4, 150)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 150)
        self.linear3 = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()
        self.vp = vp_layer_coeffs(ada_hermite, n_in=300,
                                         n_out=4, nparams=2, penalty=penalty,
                                         device=device)

    def forward(self, x):
        x = x.expand(1, x.shape[0], x.shape[1])
        x = torch.transpose(x, 0, 1)
        out = self.vp(x)
        out = torch.squeeze(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        # out = torch.round(out)

        return out


class VPNet8(nn.Module):
    def __init__(self, penalty=20.0):
        super().__init__()
        self.linear1 = nn.Linear(8, 150)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 150)
        self.linear3 = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()
        self.vp = vp_layer_coeffs(ada_hermite, n_in=300,
                                         n_out=8, nparams=2, penalty=penalty,
                                         device=device)

    def forward(self, x):
        x = x.expand(1, x.shape[0], x.shape[1])
        x = torch.transpose(x, 0, 1)
        out = self.vp(x)
        out = torch.squeeze(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        out = torch.round(out)

        return out


class VPNet12(nn.Module):
    def __init__(self, penalty=5.0):
        super().__init__()
        self.linear1 = nn.Linear(12, 150)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 150)
        self.linear3 = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()
        self.vp = vp_layer_coeffs(ada_hermite, n_in=300,
                                         n_out=12, nparams=2, penalty=penalty,
                                         device=device)

    def forward(self, x):
        x = x.expand(1, x.shape[0], x.shape[1])
        x = torch.transpose(x, 0, 1)
        out = self.vp(x)
        out = torch.squeeze(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)

        return out


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(291, 150)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 150)
        self.linear3 = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv1d(1, 1, 10, stride=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv(x)
        # add dropout layer
        out = self.relu(out)
        out = out.squeeze()
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)

        return out


class FCNNet(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.linear1 = nn.Linear(300, 150, device=device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 150, device=device)
        self.linear3 = nn.Linear(150, 1, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x.squeeze()
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)

        return out

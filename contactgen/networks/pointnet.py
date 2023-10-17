import torch
import torch.nn as nn
import torch.nn.functional as F


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, final_nl=False):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        x_out = x_s + dx
        if final_nl:
            return F.leaky_relu(x_out, negative_slope=0.2)
        return x_out


class Pointnet(nn.Module):
    ''' PointNet-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(2 * hidden_dim, 4 * hidden_dim, 1)
        self.conv4 = torch.nn.Conv1d(4 * hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.permute(0, 2, 1)

        return x, self.pool(x, dim=1)




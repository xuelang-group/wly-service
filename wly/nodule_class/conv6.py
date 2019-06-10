import numpy as np
import wly.nodule_class.config as cfg
from torch import nn


class Conv3d(nn.Module):
    def __init__(self, n_in, n_out, n=2, dropout=False):
        super(Conv3d, self).__init__()
        self.dropout = dropout
        self.n = n
        if dropout:
            self.drop3d = nn.Dropout3d(0.25)
        self.conv3d_1 = nn.Conv3d(n_in, n_out, kernel_size=3, padding=1, bias=False)
        self.bn3d_1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv3d_2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1, bias=False)
        self.bn3d_2 = nn.BatchNorm3d(n_out)
        if n==3:
            self.conv3d_3 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1, bias=False)
            self.bn3d_3 = nn.BatchNorm3d(n_out)

    def forward(self, x):
        if self.dropout:
            x = self.drop3d(x)

        x = self.conv3d_1(x)
        x = self.bn3d_1(x)
        x = self.relu(x)

        x = self.conv3d_2(x)
        x = self.bn3d_2(x)
        x = self.relu(x)

        if self.n == 3:
            x = self.conv3d_3(x)
            x = self.bn3d_3(x)
            x = self.relu(x)

        return x


class FC(nn.Module):
    def __init__(self, n_in, n_out, dropout=False, is_bn=False, is_relu=True):
        super(FC, self).__init__()
        self.dropout = dropout
        self.is_bn = is_bn
        self.is_relu = is_relu
        if dropout:
            self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(n_in, n_out, bias=(not is_bn))
        if is_bn:
            self.bn = nn.BatchNorm1d(n_out)
        if is_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.dropout:
            x = self.drop(x)
        x = self.linear(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_relu:
            x = self.relu(x)
        return x


class Net(nn.Module):

    def __init__(self, k=32, input_size=np.array(cfg.INPUT_SIZE)):
        super(Net, self).__init__()
        self.k = k
        feature_size = int((input_size / (8, 8, 4)).prod())

        self.block_1 = Conv3d(1, k, dropout=False)
        self.maxpool_1 = nn.MaxPool3d((2, 2, 1), (2, 2, 1))

        self.block_2 = Conv3d(k, k * 2, dropout=True)
        self.maxpool_2 = nn.MaxPool3d(2, 2)

        self.block_3 = Conv3d(k * 2, k * 4, dropout=True)
        self.maxpool_3 = nn.MaxPool3d(2, 2)

        self.linear_1 = FC(k * 4 * feature_size, 512, dropout=True)

        self.linear_2 = FC(512, 512, dropout=True)

        self.cls = FC(512, 2, dropout=True, is_relu=False)

    def forward(self, x):
        x = self.block_1(x)
        x = self.maxpool_1(x)

        x = self.block_2(x)
        x = self.maxpool_2(x)

        x = self.block_3(x)
        x = self.maxpool_3(x)

        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.linear_2(x)

        out = self.cls(x)
        return out

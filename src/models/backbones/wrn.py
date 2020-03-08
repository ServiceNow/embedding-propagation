# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class Block(torch.nn.Module):
    def __init__(self, ni, no, stride, dropout=0):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride=stride, padding=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(no)
        torch.nn.init.kaiming_normal_(self.conv0.weight.data)
        self.bn1 = torch.nn.BatchNorm2d(no)
        if dropout == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = torch.nn.Dropout2d(dropout)
        self.conv1 = torch.nn.Conv2d(no, no, 3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv1.weight.data)
        self.reduce = ni != no
        if self.reduce:
            self.conv_reduce = torch.nn.Conv2d(ni, no, 1, stride=stride, bias=False)
            torch.nn.init.kaiming_normal_(self.conv_reduce.weight.data)

    def forward(self, x):
        y = self.conv0(x)
        y = F.relu(self.bn0(y), inplace=True)
        y = self.dropout(y)
        y = self.conv1(y)
        y = self.bn1(y)
        if self.reduce:
            return F.relu(y + self.conv_reduce(x), True)
        else:
            return F.relu(y + x, True)


class Group(torch.nn.Module):
    def __init__(self, ni, no, n, stride, dropout=0):
        super().__init__()
        self.n = n
        for i in range(n):
            self.__setattr__("block_%d" % i, Block(ni if i == 0 else no, no, stride if i == 0 else 1, dropout=dropout))

    def forward(self, x):
        for i in range(self.n):
            x = self.__getattr__("block_%d" % i)(x)
        return x


class WideResNet(torch.nn.Module):
    def __init__(self, depth, width, exp_dict):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.n = (depth - 4) // 6
        self.output_size = 640
        self.widths = torch.Tensor([16, 32, 64]).mul(width).int().numpy().tolist()
        self.conv0 = torch.nn.Conv2d(3, self.widths[0] // 2, 3, padding=1, bias=False)
        self.bn_0 = torch.nn.BatchNorm2d(self.widths[0] // 2)
        self.dropout_prob = exp_dict["dropout"]
        self.group_0 = Group(self.widths[0] // 2, self.widths[0], self.n, 2, dropout=self.dropout_prob)
        self.group_1 = Group(self.widths[0], self.widths[1], self.n, 2, dropout=self.dropout_prob)
        self.group_2 = Group(self.widths[1], self.widths[2], self.n, 2, dropout=self.dropout_prob)
        self.bn_out = torch.nn.BatchNorm1d(self.output_size)

    def get_base_parameters(self):
        parameters = []
        parameters += list(self.conv0.parameters())
        parameters += list(self.group_0.parameters())
        parameters += list(self.group_1.parameters())
        parameters += list(self.group_2.parameters())
        parameters += list(self.bn.parameters())
        if self.embedding:
            parameters += list(self.conv_embed)
        return parameters

    def get_classifier_parameters(self):
        return self.classifier.parameters()

    def add_classifier(self, nclasses, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, nclasses))

    def forward(self, x, **kwargs):
        o = F.relu(self.bn_0(self.conv0(x)), True)
        o = self.group_0(o)
        o = self.group_1(o)
        o = self.group_2(o)
        o = o.mean(3).mean(2)
        o = F.relu(self.bn_out(o.view(o.size(0), -1)))
        return o
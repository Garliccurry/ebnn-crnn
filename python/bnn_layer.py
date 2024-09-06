"""
    BNN layer operation
    File    :bnn_layer.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :Binary network layers
"""
import math
import torch
import torch.nn as nn
from bnn_ops import Binarize, Hardtanh, LBitTanh
import torch.nn.functional as F


class BLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, binarize_input=False):
        super(BLinear, self).__init__()
        self.binarize_input = binarize_input
        self.linear = nn.Linear(in_features, out_features, bias)

        # Initialize weights and biases
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if self.binarize_input:
            x = Binarize.apply(x)
        w = Binarize.apply(self.linear.weight)
        out = F.linear(x, w, self.linear.bias)
        return out

class QLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, quantify_input=False):
        super(QLinear, self).__init__()
        self.quantify_input = quantify_input
        self.linear = nn.Linear(in_features, out_features, bias)

        # Initialize weights and biases
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if self.quantify_input:
            x = LBitTanh.apply(x)
        w = Binarize.apply(self.linear.weight)
        out = F.linear(x, w, self.linear.bias)
        return out


class BConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 binarize_input=False):
        super(BConv2d, self).__init__()
        self.binarize_input = binarize_input
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype=torch.float32)

        # Initialize weights and biases
        nn.init.xavier_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        if self.binarize_input:
            x = Binarize.apply(x)
        w = Binarize.apply(self.conv.weight)
        out = F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return out

class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 quantify_input=False):
        super(QConv2d, self).__init__()
        self.quantify_input = quantify_input
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype=torch.float32)

        # Initialize weights and biases
        nn.init.xavier_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        if self.quantify_input:
            x = LBitTanh.apply(x)
        w = Binarize.apply(self.conv.weight)
        out = F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return out

class HLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, binarize_input=False):
        super(HLinear, self).__init__()
        self.binarize_input = binarize_input
        self.linear = nn.Linear(in_features, out_features, bias, dtype=torch.float32)
        # Initialize weights and biases
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # w = F.hardtanh(self.linear.weight, min_val= -1, max_val=1)
        w = Hardtanh.apply(self.linear.weight)
        out = F.linear(x, w, self.linear.bias)
        return out

class BLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size,  use_bias=True, binarize_input=False):
        super(BLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.binarize_input = binarize_input

        self.W_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size).float())
        self.W_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size).float())
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size).float())
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size).float())

        if self.use_bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size).float())
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size).float())
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            if weight is not None:
                weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        if self.binarize_input:
            input = Binarize.apply(input)

        hx, cx = hx
        gates = (F.linear(input, Binarize.apply(self.W_ih), self.bias_ih) +
                 F.linear(hx, Binarize.apply(self.W_hh), self.bias_hh))

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, use_bias=True, bidirectional=False, binarize_input=False):
        super(BLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.binarize_input = binarize_input

        self.layers_f = nn.ModuleList([BLSTMCell(input_size if i == 0 else hidden_size,
                                                    hidden_size, use_bias, binarize_input)
                                     for i in range(num_layers)])

        if self.bidirectional:
            self.layers_r = nn.ModuleList([BLSTMCell(input_size if i == 0 else hidden_size,
                                                                hidden_size, use_bias, binarize_input)
                                                 for i in range(num_layers)])

    def forward(self, input, hx=None):
        if hx is None:
            h_zeros = torch.zeros(input.size(1), self.hidden_size, dtype=torch.float32, device=input.device)
            c_zeros = torch.zeros(input.size(1), self.hidden_size, dtype=torch.float32, device=input.device)
            hx = [(h_zeros, c_zeros) for _ in range(self.num_layers)]

        output_forward = []
        hx_forward = hx
        for t in range(input.size(0)):
            x = input[t]
            for i, layer in enumerate(self.layers_f):
                hx_forward[i] = layer(x, hx_forward[i])
                x = hx_forward[i][0]
            output_forward.append(x)

        if self.bidirectional:
            output_reverse = []
            hx_reverse = hx
            for t in reversed(range(input.size(0))):
                x = input[t]
                for i, layer in enumerate(self.layers_r):
                    hx_reverse[i] = layer(x, hx_reverse[i])
                    x = hx_reverse[i][0]
                output_reverse.insert(0, x)

            output = [torch.cat((f, r), dim=1) for f, r in zip(output_forward, output_reverse)]
        else:
            output = output_forward

        output = torch.stack(output, dim=0)
        return output, (hx_forward, hx_reverse) if self.bidirectional else hx_forward



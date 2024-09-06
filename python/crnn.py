"""
    File    :crnn.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :The main body of binary neural networks,
             BCRNN uses binary weights and fully accurate intermediate variables (activation);
             QCRNN uses binary weights and int8 quantization of intermediate variables (activation)
             quantization bits can be modified in the "Q_factor" of file "bnn_ops.py"
"""
import os
import collections

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from bnn_layer import BLinear, BConv2d, QLinear, QConv2d, HLinear, BLSTM

# path = r'data\\chinese\\size40font3shape25050'

class CaptchaDataset():
    def __init__(self, df, transform=None, path=None):
        self.df = df
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path, data['image'])).convert('L')
        label = torch.tensor(data['label'], dtype=torch.int32)

        if self.transform is not None:
            image = self.transform(image)

        return image, label



class BCRNN(nn.Module):
    def __init__(self, in_channels, output, print_val=False):
        super(BCRNN, self).__init__()

        self.print_val = print_val
        self.conv1 = BConv2d(in_channels, 4, 3, stride=1, padding=1, bias=False, binarize_input=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = BConv2d(4, 32, (4, 4), stride=1, padding=1, bias=False, binarize_input=False)
        self.pool2 = nn.MaxPool2d((4, 2), (4, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = BConv2d(32, 64, 3, stride=1, padding=1, bias=False, binarize_input=False)
        self.pool3 = nn.MaxPool2d((3, 2), (3, 2))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = BConv2d(64, 128, (3, 4), stride=1, padding=1, bias=False, binarize_input=False)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 2))
        self.bn4 = nn.BatchNorm2d(128)

        self.linear1 = BLinear(128, 64, bias=False, binarize_input=False)
        self.rnn = BLSTM(64, 128, use_bias=False, bidirectional=True)
        self.bn6 = nn.BatchNorm1d(256)
        self.linear2 = HLinear(256, output + 1, bias=False)

    def forward(self, X, y=None, criterion=None):  # [16, 1, 50, 250]
        out1 = self.conv1(X)        # [16, 1, 50, 250]
        out1 = self.pool1(out1)     # [16, 4, 25, 125]
        out1 = self.bn1(out1)
        out2 = self.conv2(out1)     # [16, 4, 24, 124]
        out2 = self.pool2(out2)     # [16, 32, 6, 62]
        out2 = self.bn2(out2)
        out3 = self.conv3(out2)     # [16, 32, 6, 62]
        out3 = self.pool3(out3)     # [16, 128, 2, 31]
        out3 = self.bn3(out3)
        out4 = self.conv4(out3)     # [16, 128, 2, 31]
        out4 = self.pool4(out4)     # [16, 256, 1, 15]
        out4 = self.bn4(out4)

        N, C, h, w = out4.size()
        out4 = out4.view(N, -1, w)      # [16, 256*1, 31]   N C*H W
        out5 = out4.permute(0, 2, 1)    # [16, 31, 256]     N W C*H
        out6 = self.linear1(out5)       # [16, 31, 256]     N W K

        out7 = out6.permute(1, 0, 2)    # [31, 16, 128]     W N K
        out8, _ = self.rnn(out7)        # [31, 16, 256]     W N K
        out9 = out8.permute(1, 2, 0)    # [16, 256, 31]     W K N
        out9 = self.bn6(out9)
        out9 = out9.permute(2, 0, 1)    # [31, 16, 256]     W N K
        out = self.linear2(out9)        # [31, 16, y]       W N K
        if y is not None:
            T = out.size(0)
            N = out.size(1)

            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)

            loss = criterion(out, y, input_lengths, target_lengths)

            return out, loss

        return out, None

class QCRNN(nn.Module):
    def __init__(self, in_channels, output, print_val=False):
        super(QCRNN, self).__init__()

        self.print_val = print_val
        self.conv1 = QConv2d(in_channels, 4, 3, stride=1, padding=1, bias=False, quantify_input=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = QConv2d(4, 32, (4, 4), stride=1, padding=1, bias=False, quantify_input=True)
        self.pool2 = nn.MaxPool2d((4, 2), (4, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = QConv2d(32, 128, 3, stride=1, padding=1, bias=False, quantify_input=True)
        self.pool3 = nn.MaxPool2d((3, 2), (3, 2))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = QConv2d(128, 512, (3, 4), stride=1, padding=1, bias=False, quantify_input=True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 2))
        self.bn4 = nn.BatchNorm2d(512)

        self.linear1 = QLinear(512, 64, bias=False, quantify_input=True)

        self.rnn = BLSTM(64, 128, use_bias=False, bidirectional=True)
        self.bn5 = nn.BatchNorm1d(256)
        self.linear2 = HLinear(256, output + 1, bias=False)

    def forward(self, X, y=None, criterion=None):  # [16, 1, 50, 250]
        out1 = self.conv1(X)        # [16, 1, 50, 250]
        out1 = self.pool1(out1)     # [16, 4, 25, 125]
        out1 = self.bn1(out1)
        out2 = self.conv2(out1)     # [16, 4, 24, 124]
        out2 = self.pool2(out2)     # [16, 32, 6, 62]
        out2 = self.bn2(out2)
        out3 = self.conv3(out2)     # [16, 32, 6, 62]
        out3 = self.pool3(out3)     # [16, 128, 2, 31]
        out3 = self.bn3(out3)
        out4 = self.conv4(out3)     # [16, 128, 2, 31]
        out4 = self.pool4(out4)     # [16, 512, 1, 15]
        out4 = self.bn4(out4)

        N, C, h, w = out4.size()
        out4 = out4.view(N, -1, w)      # [16, 512*1, 15]   N C*H W
        out5 = out4.permute(0, 2, 1)    # [16, 15, 256]     N W C*H
        out6 = self.linear1(out5)       # [16, 15, 256]     N W K

        out7 = out6.permute(1, 0, 2)    # [15, 16, 128]     W N K
        out8, _ = self.rnn(out7)        # [15, 16, 256]     W N K
        out9 = out8.permute(1, 2, 0)    # [16, 256, 15]     W K N
        out9 = self.bn5(out9)
        out9 = out9.permute(2, 0, 1)    # [15, 16, 256]     W N K
        out = self.linear2(out9)        # [15, 16, y]       W N K

        if y is not None:
            T = out.size(0)
            N = out.size(1)

            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)

            loss = criterion(out, y, input_lengths, target_lengths)

            return out, loss

        return out, None

class Engine:
    def __init__(self, model, optimizer, criterion, epochs=50, early_stop=False, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device

    def fit(self, dataloader):
        hist_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                self.optimizer.zero_grad()

                out, loss = self.model(data, target, criterion=self.criterion)

                loss.backward()

                self.optimizer.step()

                tk.set_postfix({'Epoch': epoch + 1, 'Loss': loss.item()})

    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        hist_loss = []
        outs = collections.defaultdict(list)
        tk = tqdm(dataloader, total=len(dataloader))
        with torch.no_grad():
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                out, loss = self.model(data, target, criterion=self.criterion)

                target = target.cpu().detach().numpy()
                outs['pred'].append(out)
                outs['target'].append(target)

                hist_loss.append(loss)

                tk.set_postfix({'Loss': loss.item()})

        return outs, hist_loss

    def predict(self, image):
        image = Image.open(image).convert('L')
        image_tensor = T.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0)
        out, _ = self.model(image_tensor.to(device=self.device))
        out = out.permute(1, 0, 2)
        out = out.log_softmax(2)
        out = out.argmax(2)
        out = out.cpu().detach().numpy()

        return out

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['mapping'], checkpoint['mapping_inv']

def calculate_loss(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs, loss = model(images, labels, criterion)
            # 累加损失
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    average_loss = total_loss / total_samples
    return average_loss

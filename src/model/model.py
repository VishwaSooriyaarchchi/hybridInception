# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - TorchVision

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


class TargetNet(nn.Module):
    """ TargetNet for microRNA target prediction """
    def __init__(self, model_cfg, with_esa, dropout_rate):
        super(TargetNet, self).__init__()
        num_channels = model_cfg.num_channels
        num_blocks = model_cfg.num_blocks

        if not with_esa: self.in_channels, in_length = 8, 40
        else:            self.in_channels, in_length = 10, 50
        out_length = np.floor(((in_length - model_cfg.pool_size) / model_cfg.pool_size) + 1)

        self.stem = self._make_layer(model_cfg, num_channels[0], num_blocks[0], dropout_rate, stem=True)
        self.stage1 = self._make_layer(model_cfg, num_channels[1], num_blocks[1], dropout_rate)
        self.stage2 = self._make_layer(model_cfg, num_channels[2], num_blocks[2], dropout_rate)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.inception = InceptionWithAttention(10, 16, (16, 32), (8, 16), 12)
        self.max_pool = nn.MaxPool1d(model_cfg.pool_size)
        # self.linear = nn.Linear(int(num_channels[-1] * out_length), 1)
        self.linear = nn.Linear(512, 1)

        #changes for the fused gate parts
        self.inception_proj = nn.Linear(3800, 512)
        self.feature_dim = 512  # Replace F with the correct dimension of each branch's flattened output.
        self.fusion_gate = FusionGate(self.feature_dim)

    def _make_layer(self, cfg, out_channels, num_blocks, dropout_rate, stem=False):
        layers = []
        for b in range(num_blocks):
            if stem: layers.append(Conv_Layer(self.in_channels, out_channels, cfg.stem_kernel_size, dropout_rate,
                                              post_activation= b < num_blocks - 1))
            else:    layers.append(ResNet_Block(self.in_channels, out_channels, cfg.block_kernel_size, dropout_rate,
                                                skip_connection=cfg.skip_connection))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x2 = self.inception(x)
        # Project the Inception output to the common dimension (512)
        x2 = self.inception_proj(x2)

        x = self.stem(x)
        # print(x.shape)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dropout(self.relu(x))
        x = self.max_pool(x)
        # print(f"ResNet output shape: {x.shape}")
        print(f"Inception output shape: {x2.shape}")
        x = x.reshape(len(x), -1)
        print(f"Flattened ResNet output shape: {x.shape}")
        # x = torch.cat((x,x2), dim = 1)

        # Use the FusionGate to combine features:
        x = self.fusion_gate(x, x2)

        x = self.dropout(self.relu(x))
        # print(f"Concatenated output shape: {x.shape}")
        x = self.linear(x)
        # print(x.shape,'here ', x)

        return x


def conv_kx1(in_channels, out_channels, kernel_size, stride=1):
    """ kx1 convolution with padding without bias """
    layers = []
    padding = kernel_size - 1
    padding_left = padding // 2
    padding_right = padding - padding_left
    layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False))
    return nn.Sequential(*layers)



class InceptionWithAttention(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionWithAttention, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            AttentionConvBlock(in_channels, c2[0], kernel_size=1),
            AttentionConvBlock(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            AttentionConvBlock(in_channels, c3[0], kernel_size=1),
            AttentionConvBlock(c3[0], c3[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, c4, kernel_size=1),
            nn.ReLU()
        )
        self.ca = ChannelAttention(c1 + c2[1] + c3[1] + c4)
        self.sa = SpatialAttention()
        self.flatten = nn.Flatten()

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        out = torch.cat((p1, p2, p3, p4), dim=1)
        out = self.ca(out)
        out = self.sa(out)
        out = self.flatten(out)

        return out



class AttentionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(AttentionConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        attention_weights = self.attention(out)
        out = out * attention_weights
        out = self.relu(out)
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class Conv_Layer(nn.Module):
    """
    CNN layer with/without activation
    -- Conv_kx1_ReLU-Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, post_activation):
        super(Conv_Layer, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.conv = conv_kx1(in_channels, out_channels, kernel_size)
        self.post_activation = post_activation

    def forward(self, x):
        out = self.conv(x)
        if self.post_activation:
            out = self.dropout(self.relu(out))

        return out


class ResNet_Block(nn.Module):
    """
    ResNet Block
    -- ReLU-Dropout-Conv_kx1 - ReLU-Dropout-Conv_kx1
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, skip_connection):
        super(ResNet_Block, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.conv1 = conv_kx1(in_channels, out_channels, kernel_size)
        self.conv2 = conv_kx1(out_channels, out_channels, kernel_size)
        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.dropout(self.relu(x))
        out = self.conv1(out)

        out = self.dropout(self.relu(out))
        out = self.conv2(out)

        if self.skip_connection:
            out_c, x_c = out.shape[1], x.shape[1]
            if out_c == x_c: out += x
            else:            out += F.pad(x, (0, 0, 0, out_c - x_c))

        return out

#learnable fusion gated layer
class FusionGate(nn.Module):
    def __init__(self, feature_dim):
        super(FusionGate, self).__init__()
        self.linear = nn.Linear(feature_dim * 2, feature_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feat_resnet, feat_inception):
        combined = torch.cat([feat_resnet, feat_inception], dim=1)
        gate = self.sigmoid(self.linear(combined))
        fused_features = gate * feat_resnet + (1 - gate) * feat_inception
        return fused_features

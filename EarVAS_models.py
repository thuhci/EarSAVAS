import math
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

class FocalLossMulti(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean', eps=1e-10):
        super(FocalLossMulti, self).__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        # print(targets.device)
        self.alpha = self.alpha.to(targets.device)
        targets_one_hot = F.one_hot(targets, num_classes=len(self.alpha))
        targets_one_hot = targets_one_hot.float()

        # 计算交叉熵损失
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 计算概率值
        pt = torch.exp(-CE_loss)
        # 使用clamp限制概率的范围以提高数值稳定性
        pt = torch.clamp(pt, min=self.eps, max=1-self.eps)
        # 计算Focal Loss
        alpha_t = (self.alpha * targets_one_hot).sum(dim=1)
        
        F_loss = alpha_t * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class MeanPooling(nn.Module):
    def __init__(self, n_in, n_out, cla_activation):
        super(MeanPooling, self).__init__()
        self.cla_activation = cla_activation
        self.cla = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.cla)

    def activate(self, x, activation):
        if activation == 'None':
            return x
        return torch.sigmoid(x)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)
        cla = cla[:, :, :, 0]
        x = torch.mean(cla, dim=2)
        return x

class IMUFeatureExtractor(nn.Module):
    def __init__(self, output_dim, samosa=False):
        super(IMUFeatureExtractor, self).__init__()
        self.output_dim = output_dim
        self.samosa = samosa
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=128, kernel_size=10, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=10, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=10, stride=1, padding=0)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.maxpool1d2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2816, self.output_dim)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.maxpool1d(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.maxpool1d2(x)

        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        if not self.samosa:
            x = self.fc1(x)
        return x

class AudioFeatureExtractorSAMOSA(nn.Module):
    def __init__(self, in_channels=1):
        super(AudioFeatureExtractorSAMOSA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool2d1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2d2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool2d3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2d4(x)

        x = x.flatten(start_dim=1)        
        return x

class AudioFeatureExtractor(nn.Module):
    def __init__(self, label_dim=527, in_channels=1):
        super(AudioFeatureExtractor, self).__init__()
        self.effnet = EfficientNet.from_name('efficientnet-b0', in_channels=in_channels)
        self.attention = MeanPooling(
            1280,
            label_dim,
            cla_activation='None')
        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()

    def forward(self, x):    
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = x.transpose(2, 3)
        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2, 3)
        out = self.attention(x)
        return out

class EarVAS(nn.Module):
    def __init__(self, num_classes, fusion=True, audio_channel='BiChannel', feature_size=256, samosa=False) -> None:
        super(EarVAS, self).__init__()
        self.fusion = fusion
        self.audio_channel = audio_channel
        self.samosa = samosa

        if samosa and audio_channel != 'None':
            self.audio_feature_extractor = AudioFeatureExtractorSAMOSA(in_channels=2) if \
                audio_channel == 'BiChannel' else AudioFeatureExtractorSAMOSA(in_channels=1)
        elif not samosa and audio_channel != 'None':
            self.audio_feature_extractor = AudioFeatureExtractor(label_dim=feature_size, in_channels=2) if \
                audio_channel == 'BiChannel' else AudioFeatureExtractor(label_dim=feature_size, in_channels=1)

        if fusion or audio_channel == 'None':
            self.imu_feature_extractor = IMUFeatureExtractor(output_dim=feature_size, samosa=samosa)
        
        if samosa:
            if fusion:
                self.fc1 = nn.Linear(15104, 1000)
            elif audio_channel == 'BiChannel' or audio_channel == 'FeedBack' or audio_channel == 'FeedForward':
                self.fc1 = nn.Linear(12288, 1000)
            elif audio_channel == 'None':
                self.fc1 = nn.Linear(2816, 1000)
            self.fc2 = nn.Linear(1000, 500)
            self.fc3 = nn.Linear(500, 250)
            self.fc_recognition = nn.Linear(250, num_classes)
        else:
            self.fusion_fc = nn.Linear(2 * feature_size, feature_size) if \
                fusion else nn.Linear(feature_size, feature_size)
            self.fc_recognition = nn.Linear(feature_size, num_classes)

    def forward(self, audio, imu):
        if self.fusion:
            if self.audio_channel == 'BiChannel':
                audio_features = self.audio_feature_extractor(audio)
            elif self.audio_channel == 'FeedBack':
                audio_features = self.audio_feature_extractor(audio[:, 0, :, :])
            elif self.audio_channel == 'FeedForward':
                audio_features = self.audio_feature_extractor(audio[:, 1, :, :])
            else:
                raise ValueError('audio channel is not supported')
            imu_features = self.imu_feature_extractor(imu)
            features = torch.cat((audio_features, imu_features), dim=1)
            if self.samosa:
                features = self.fc1(features)
                features = F.relu(features)
                features = self.fc2(features)
                features = F.relu(features)
                features = self.fc3(features)
                features = F.relu(features)
            else:
                features = self.fusion_fc(features)
                features = F.relu(features)
            res = self.fc_recognition(features)
        else:
            if self.audio_channel == 'BiChannel':
                audio_features = self.audio_feature_extractor(audio)
                if self.samosa:
                    audio_features = self.fc1(audio_features)
                    audio_features = F.relu(audio_features)
                    audio_features = self.fc2(audio_features)
                    audio_features = F.relu(audio_features)
                    audio_features = self.fc3(audio_features)
                    audio_features = F.relu(audio_features)
                else:
                    audio_features = self.fusion_fc(audio_features)
                    audio_features = F.relu(audio_features)
                res = self.fc_recognition(audio_features)
            elif self.audio_channel == 'FeedBack':
                audio_features = self.audio_feature_extractor(audio[:, 0, :, :])
                if self.samosa:
                    audio_features = self.fc1(audio_features)
                    audio_features = F.relu(audio_features)
                    audio_features = self.fc2(audio_features)
                    audio_features = F.relu(audio_features)
                    audio_features = self.fc3(audio_features)
                    audio_features = F.relu(audio_features)
                else:
                    audio_features = self.fusion_fc(audio_features)
                    audio_features = F.relu(audio_features)
                res = self.fc_recognition(audio_features)
            elif self.audio_channel == 'FeedForward':
                audio_features = self.audio_feature_extractor(audio[:, 1, :, :])
                if self.samosa:
                    audio_features = self.fc1(audio_features)
                    audio_features = F.relu(audio_features)
                    audio_features = self.fc2(audio_features)
                    audio_features = F.relu(audio_features)
                    audio_features = self.fc3(audio_features)
                    audio_features = F.relu(audio_features)
                else:
                    audio_features = self.fusion_fc(audio_features)
                    audio_features = F.relu(audio_features)
                res = self.fc_recognition(audio_features)
            elif self.audio_channel == 'None':
                imu_features = self.imu_feature_extractor(imu)
                if self.samosa:
                    imu_features = self.fc1(imu_features)
                    imu_features = F.relu(imu_features)
                    imu_features = self.fc2(imu_features)
                    imu_features = F.relu(imu_features)
                    imu_features = self.fc3(imu_features)
                    imu_features = F.relu(imu_features)
                else:
                    imu_features = self.fusion_fc(imu_features)
                    imu_features = F.relu(imu_features)
                res = self.fc_recognition(imu_features)
            else:
                raise ValueError('audio channel is not supported')
        return res

class EffNetMean(nn.Module):
    def __init__(self, label_dim=527):
        super(EffNetMean, self).__init__()
        self.effnet = EfficientNet.from_name('efficientnet-b0', in_channels=1)
        self.attention = MeanPooling(
            1280,
            label_dim,
            cla_activation='None')
        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = x.transpose(2, 3)
        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2, 3)
        out = self.attention(x)
        return out
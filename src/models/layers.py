import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3
from src.constant import VIEWS


class Conv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv2dLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MaxPoolLayer(nn.Module):

    def __init__(self, height, width):
        super(MaxPoolLayer, self).__init__()
        self.pool = nn.MaxPool2d(height, width)

    def forward(self, x):
        return self.pool(x)


class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)

    def forward(self, x):
        h = self.fc_layer(x)
        if len(self.output_shape) > 1:
            h = h.view(h.shape[0], *self.output_shape)
        h = F.log_softmax(h, dim=-1)
        return h


class BasicBlockV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        return {
            VIEWS.L_CC: self.single_add_gaussian_noise(x[VIEWS.L_CC]),
            VIEWS.L_MLO: self.single_add_gaussian_noise(x[VIEWS.L_MLO]),
            VIEWS.R_CC: self.single_add_gaussian_noise(x[VIEWS.R_CC]),
            VIEWS.R_MLO: self.single_add_gaussian_noise(x[VIEWS.R_MLO]),
        }

    def single_add_gaussian_noise(self, single_view):
        if not self.gaussian_noise_std or not self.training:
            return single_view
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            view_name: self.single_avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }

    @staticmethod
    def single_avg_pool(single_view):
        n, c, _, _ = single_view.size()
        return single_view.view(n, c, -1).mean(-1)

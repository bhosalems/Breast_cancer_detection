import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid
import numpy as np
from src.constant import VIEWS

class Spatial_attn(nn.Module):

    def __init__(self, planes, N):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1_1 = nn.Conv2d(planes, planes//8, 1)
        self.conv1x1_2 = nn.Conv2d(planes, planes//8, 1)
        self.conv1x1_3 = nn.Conv2d(planes, planes//8, 1)

    
    def forward(self,x):
        Conv2d_1 = self.conv1x1_1(x)
        Conv2d_2 = self.conv1x1_2(x)
        Conv2d_3 = self.conv1x1_3(x)

        Conv2d_1_flatten = torch.flatten(Conv2d_1, start_dim=2, end_dim=3)
        Conv2d_2_flatten = torch.flatten(Conv2d_2, start_dim=2, end_dim=3)
        # Conv2d_2_flatten_tranpose = Conv2d_2_flatten.transpose(2, 1)
        Conv2d_2_flatten = Conv2d_2_flatten.transpose(2, 1)

        # similarity_matrix = Conv2d_2_flatten_tranpose @ Conv2d_1_flatten
        # similarity_matrix_softmax = self.softmax(similarity_matrix)

        # Conv2d_3_flatten = torch.flatten(Conv2d_3,start_dim=2, end_dim=3)

        # features = Conv2d_3_flatten @ similarity_matrix_softmax.permute(0,2,1)

        # features_sigmoid = self.sigmoid(features)
        
        # features_sigmoid_exp = features_sigmoid.reshape((x.size()[0], 1, x.size()[-2], x.size()[-1]))
        # out = x * features_sigmoid_exp
        Conv2d_2_flatten = Conv2d_2_flatten @ Conv2d_1_flatten
        Conv2d_2_flatten = self.softmax(Conv2d_2_flatten)

        Conv2d_3_flatten = torch.flatten(Conv2d_3,start_dim=2, end_dim=3)

        Conv2d_3_flatten = Conv2d_3_flatten @ Conv2d_2_flatten.permute(0,2,1)

        Conv2d_3_flatten = self.sigmoid(Conv2d_3_flatten)
        
        Conv2d_3_flatten = Conv2d_3_flatten.reshape((x.size()[0], 1, x.size()[-2], x.size()[-1]))
        Conv2d_3_flatten = x * Conv2d_3_flatten

        return Conv2d_3_flatten

class Channel_attn(nn.Module):

    def __init__(self, planes, N):
        super().__init__()
        self.N = N
        self.dense1 = nn.Linear(planes, N)
        self.dense2 = nn.Linear(planes, N)
        self.dense3 = nn.Linear(planes, N)
        self.softmax = nn.Softmax(dim=2)
        self.dense_second = nn.Linear(N, planes)
        self.sigmoid = nn.Sigmoid()

    def global_average_pooling(self, x):
        gap = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        gap_reshaped = gap.reshape(gap.shape[0], 1, gap.shape[1])
        return gap_reshaped

    def forward(self, x):
        #Global Average Pooling
        x_gap = self.global_average_pooling(x)
        dense1 = self.dense1(x_gap)
        dense2 = self.dense2(x_gap)
        dense3 = self.dense3(x_gap)

        dense1 = dense1.transpose(2, 1)
        # print(dense1.size())
        dense2 = dense1 @ dense2
        dense2 = self.softmax(dense2)
        # print("dense 3",dense3.size())
        # print("similarity matrix", dense2.size())
        dense3 = dense3 @ dense2.permute(0,2,1)
        # print("dense 3 ", dense3.size())

        dense3 = self.dense_second(dense3)
        adding_layer = x_gap + dense3
        sigmoid = self.sigmoid(adding_layer)
        
        sigmoid = sigmoid.transpose(2,1)
        sigmoid = sigmoid[:, :, :, None]
        sigmoid = x * sigmoid
        # print("here")
        return sigmoid
    
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

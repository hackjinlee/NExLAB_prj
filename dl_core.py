import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import enum

E = 0.0001 # EPSILON


class NetTypes1D(enum.Enum):
    CNN1D = 'CNN1D'
    Resnet = 'ResNet1D'
    ConvLSTM = 'ConvLSTM'
    InceptionNet = 'Inception1D'


class DeepLearningModel1D(nn.Module):
    def __init__(self, network_type, input_shape, output_size, depth, conv_channels):
        '''
        CNN 기반 딥러닝 모델 생성
        :param network_type: 네트워크 타입
        :param input_shape: 입력 데이터 형태
        :param output_size: 출력 데이터 크기
        :param depth: 네트워크 깊이 (layer 개수)
        :param conv_channels: convolution filter 개수
        '''
        super().__init__()
        self.network_type = network_type
        self.input_shape = input_shape
        self.output_size = output_size
        self.depth = depth
        self.conv_channels = conv_channels

        if network_type == NetTypes1D.Resnet:
            model = ResNet(input_shape, output_size, depth, conv_channels)
        elif network_type == NetTypes1D.ConvLSTM:
            model = ConvLSTM(input_shape, output_size, depth, conv_channels)
        elif network_type == NetTypes1D.InceptionNet:
            model = InceptionNet(input_shape, output_size, depth, conv_channels)
        elif network_type == NetTypes1D.CNN1D:
            model = CNN1D(input_shape, output_size, depth, conv_channels)
        else:
            raise NotImplementedError

        self.model = model
        # weight initialization
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.model(x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv_kxk = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=E)

    def forward(self, x):
        x = self.conv_kxk(x)
        x = self.bn(x)
        out = F.leaky_relu(x, inplace=True)
        return out


class CNN1D(nn.Module):
    DR = 0.25

    def __init__(self, input_shape, output_size, depth, conv_channels):
        super(CNN1D, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.depth = depth
        self.conv_channels = conv_channels

        self.init_block = ConvBlock(input_shape[0], conv_channels, kernel_size=5, padding=2)

        block_dict = dict()
        for d in range(1, depth):
            block = ConvBlock(conv_channels, conv_channels, kernel_size=5, padding=2)
            name = 'Block%s' % d
            self.add_module(name, block)
            block_dict[name] = block
        self.block_dict = block_dict

        # linear
        self.dropout = nn.Dropout(p=self.DR)
        self.linear = nn.Linear(conv_channels, output_size, bias=True)
        self.output_size = output_size

    def forward(self, x):
        b_out = self.init_block(x)

        for d in range(1, self.depth):
            name = 'Block%s' % d
            block = self.block_dict[name]
            b_out = F.max_pool1d(b_out, kernel_size=2, padding=0)
            b_out = block(b_out)

        ap = F.avg_pool1d(b_out, b_out.size(2)).squeeze(2)
        do = self.dropout(ap)
        out = self.linear(do)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm1d(out_channels, eps=E)
        self.conv2 = nn.Conv1d(out_channels, out_channels, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm1d(out_channels, eps=E)

        self.use_residual = True if in_channels == out_channels else False

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_residual:
            x += residual
        out = F.leaky_relu(x)
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, output_size, depth, conv_channels):
        super(ResNet, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.depth = depth
        self.conv_channels = conv_channels

        self.init_block = ResidualBlock(input_shape[0], conv_channels, kernel_size=5, padding=2)

        block_dict = dict()
        for d in range(1, depth):
            block = ResidualBlock(conv_channels, conv_channels, kernel_size=5, padding=2)
            name = 'Block%s' % d
            self.add_module(name, block)
            block_dict[name] = block
        self.block_dict = block_dict

        # linear
        self.dropout = nn.Dropout(p=0.25)
        self.linear = nn.Linear(conv_channels, output_size, bias=True)
        self.output_size = output_size


    def forward(self, x):
        b_out = self.init_block(x)

        for d in range(1, self.depth):
            name = 'Block%s' % d
            block = self.block_dict[name]
            b_out = F.max_pool1d(b_out, kernel_size=2, padding=0)
            b_out = block(b_out)

        ap = F.avg_pool1d(b_out, b_out.size(2)).squeeze(2)
        do = self.dropout(ap)
        out = self.linear(do)
        return out


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, output_size, depth, conv_channels):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.depth = depth
        self.conv_channels = conv_channels

        self.init_block = ConvBlock(input_shape[0], conv_channels, kernel_size=5, padding=2)

        block_dict = dict()
        for d in range(1, depth):
            block = ConvBlock(conv_channels, conv_channels, kernel_size=5, padding=2)
            name = 'Block%s' % d
            self.add_module(name, block)
            block_dict[name] = block
        self.block_dict = block_dict

        last_size = input_shape[1] // (2 ** (depth-1))
        self.lstm = nn.LSTM(last_size, conv_channels, num_layers=2,
                            bias=True, batch_first=True, dropout=0.15, bidirectional=True)
        # linear
        linear_ch = conv_channels * 2
        self.dropout = nn.Dropout(p=0.25)
        self.linear = nn.Linear(linear_ch, output_size)
        self.output_size = output_size

    def forward(self, x):
        b_out = self.init_block(x)

        for d in range(1, self.depth):
            name = 'Block%s' % d
            block = self.block_dict[name]
            b_out = F.max_pool1d(b_out, kernel_size=2, padding=0)
            b_out = block(b_out)

        l_out, _ = self.lstm(b_out)
        l_out = torch.transpose(l_out, 1, 2)
        l_out = F.avg_pool1d(l_out, l_out.size(2)).squeeze(2)

        out = self.linear(l_out)
        return out


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch3_1, ch3_3, ch5_2, ch5_3):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, ch3_1, kernel_size=3, padding=1)
        self.branch2 = ConvBlock(in_channels, ch3_3, kernel_size=3, padding=3, dilation=3)
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, ch5_2, kernel_size=1),
            ConvBlock(ch5_2, ch5_2, kernel_size=5, padding=4, dilation=2)
        )
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, ch5_3, kernel_size=1),
            ConvBlock(ch5_3, ch5_3, kernel_size=5, padding=6, dilation=3)
        )
        self.use_residual = True if in_channels == ch3_1 + ch3_3 + ch5_2 + ch5_3 else False

    def forward(self, x):
        residual = x
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        c1 = torch.cat([branch1, branch2, branch3, branch4], 1)
        if self.use_residual:
            c1 += residual
        return c1


class InceptionNet(nn.Module):
    def __init__(self, input_shape, output_size, depth, conv_channels):
        super(InceptionNet, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.depth = depth
        self.conv_channels = conv_channels
        self.start_epoch = 0
        num_channels = input_shape[0]

        branch_channels = conv_channels // 4
        self.init_block = ConvBlock(input_shape[0], conv_channels, kernel_size=5, padding=2)

        block_dict = dict()
        for d in range(1, depth):
            block = InceptionBlock(conv_channels, branch_channels, branch_channels, branch_channels, branch_channels)
            name = 'Block%s' % d
            self.add_module(name, block)
            block_dict[name] = block
        self.block_dict = block_dict

        # linear
        self.dropout = nn.Dropout(p=0.25)
        self.linear = nn.Linear(conv_channels, output_size)

    def forward(self, x):
        b_out = self.init_block(x)

        for d in range(1, self.depth):
            name = 'Block%s' % d
            block = self.block_dict[name]
            b_out = F.max_pool1d(b_out, kernel_size=2, padding=0)
            b_out = block(b_out)

        # avg_pool linear
        ap = F.avg_pool1d(b_out, b_out.size(2)).squeeze(2)
        do = self.dropout(ap)
        out = self.linear(do)
        return out
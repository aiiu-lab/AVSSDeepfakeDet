import math
import torch
import torch.nn as nn
from einops import rearrange

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu", reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if relu_type == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu = nn.PReLU(num_parameters=planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEResnet(nn.Module):
    def __init__(self, layers, num_filters, **kwargs):
        super(SEResnet, self).__init__()

        block = SEBasicBlock
        self.inplanes   = num_filters[0]

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(1, 2), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))
        out_dim = num_filters[3] * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = torch.mean(x, dim=-1)
        x = rearrange(x, 'b d t -> b t d')
        # print(x.shape)
        # x = x.view((x.size()[0], x.size()[1], -1))
        # print(x.shape)
        # x = x.transpose(1, 2)
        # print(x.shape)

        return x

# class VGG(nn.Module):
#     def __init__(self, **kwargs):
#         super(VGG, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#             nn.BatchNorm2d(192),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(3,3), stride=(2,1)),
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(192, 384, kernel_size=(3,3), padding=(2,1)),
#             nn.BatchNorm2d(384),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=(1,5), padding=(0,0)),
#         )

#         # self.mlp_head = nn.Sequential(
#         #     nn.ReLU(),
#         #     nn.Linear(512, 512)
#         # )

#     def forward(self, x):
#         # out = self.net(x)
#         x = self.layer1(x)
#         # print(x.shape)
#         x = self.layer2(x)
#         # print(x.shape)
#         x = self.layer3(x)
#         # print(x.shape)
#         x = self.layer4(x)
#         # print(x.shape)
#         x = torch.squeeze(x)
#         # print(x.shape)
#         x = x.transpose(1, 2)
#         # print(x.shape)
#         # x = self.mlp_head(x)
#         # print(x.shape)
#         return x

class VGG(nn.Module):
    def __init__(self, last_dim=256, last_avg=False, temporal_half=False, **kwargs):
        super(VGG, self).__init__()
        self.last_avg = last_avg

        num_filters = [last_dim//8, last_dim//4, last_dim//2, last_dim]

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2,1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=(2,1)),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=(1,5), padding=0),
            nn.Flatten(2, 3),
        )

        # if self.last_avg:
        #     self.mlp_head = nn.Sequential(
        #         nn.Flatten(),
        #         nn.ReLU(),
        #         nn.Linear(last_dim, last_dim),
        #     )
        # else:
        #     self.mlp_head = nn.Sequential(
        #         nn.ReLU(),
        #         nn.Linear(last_dim, last_dim),
        #     )

        if temporal_half:
            self.temporal_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
        else:
            self.temporal_pool = nn.Identity()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = x.transpose(1, 2)
        x = rearrange(x, 'b c t -> b t c')
        x = self.temporal_pool(x)
        if self.last_avg:
            x = torch.mean(x, 1)
        # x = self.mlp_head(x)

        return x

class VGG2(nn.Module):
    def __init__(self, last_dim, **kwargs):
        super(VGG2, self).__init__()
        self.last_dim = last_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,1), padding=(0,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, self.last_dim, kernel_size=(5,1), padding=(0,0)),
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.last_dim),
            nn.Linear(self.last_dim, self.last_dim)
        )

    def forward(self, x):
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = torch.squeeze(x, 2)
        print(x.shape)
        x = x.transpose(1, 2)
        print(x.shape)
        x = self.mlp_head(x)
        print(x.shape)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu"):
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception("relu type not implemented")
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[1, 2, 2, 2, 2], inplanes=1, frames_per_clip=25, num_filters=[16, 32, 64, 128, 256], strides=[1, 1, (2, 1), (2, 1), (2, 2)], relu_type="relu", gamma_zero=False, avg_pool_downsample=False):
        super(ResNet, self).__init__()
        
        self.inplanes = inplanes
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        self.layers = nn.ModuleList()
        for i in range(5):
            self.layers.append(self._make_layer(block, num_filters[i], layers[i], stride=strides[i]))
        # self.layer1 = self._make_layer(block, num_filters[0], layers[0], stride=strides[0])
        # self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=strides[1])
        # self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=strides[2])
        # self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=strides[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, frames_per_clip))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_filters[-1]),
            nn.Linear(num_filters[-1], num_filters[-1])
        )

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes, outplanes=planes * block.expansion, stride=stride
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        for i in range(5):
            x = self.layers[i](x)
        x = self.avgpool(x)
        x = torch.squeeze(x, 2)
        x = x.transpose(1, 2)
        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    # 1: 63, 0.6: 38, 0.2: 13
    x = torch.rand((80, 1, 64, 63))
    # m = SEResnet(layers = [2, 2, 2, 2],  num_filters = [64, 128, 256, 512])
    # m = VGG2(last_dim=256)
    m = ResNet(inplanes=1, frames_per_clip=25)
    print('total model parameters: ', sum([p.numel() for p in m.parameters()]))
    o = m(x)
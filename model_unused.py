import torch
import torch.nn.functional as F
import math
from torch import nn

"""
class BI_CONVLSTM(nn.Module):
    def __init__(self,ins,outs):

        super(BI_CONVLSTM,self).__init__()

        self.fconv = ConvLSTM(ins,outs)
        self.bconv = ConvLSTM(ins,outs)

    def forward(self,x,s):
        xs = x.clone()
        f = self.fconv(x,time = s)

        batch,time_step,c,h,w  = x.size()

        for idx in range(batch):
            xs[idx,s[idx]:] = xs[idx,s[idx]:].flip(0)

        b = self.bconv(xs,time = s)

        return torch.cat((f,b),dim=1)

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes//2))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes//2))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes//2))
        self.trans3 = Transition(num_planes, out_planes,last = True)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.relu(self.bn(out))
        return out

def miniDensenet():
    return DenseNet(DenseBlock, [6, 12, 24, 16], growth_rate=2)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# resnet: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py and https://github.com/zhunzhong07/Random-Erasing/blob/master/models/cifar/resnet.py
class ResNet(nn.Module):
    def __init__(self, depth):
        super(ResNet, self).__init__()

        assert depth  % 6 == 0, 'depth should be 6n'
        n = depth // 6

        block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.relu(x)    # 48x48

        x = self.layer1(x)  # 48x48
        x = self.layer2(x)  # 24x24
        x = self.layer3(x)  # 12x12

        return x

def resnet12(**kwargs):
    return ResNet(12)



#AlexNet: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class miniAlexNet(nn.Module):
    def __init__(self):
        super(miniAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding = 1),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        return x



#DenseNet: https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
class DenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out



class Transition(nn.Module):
    def __init__(self, in_planes, out_planes,last = False):
        super(Transition, self).__init__()
        self.bn   = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.last = last

    def forward(self, x):
        stride = self.last and 1 or 2
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2,stride)
        return out
"""
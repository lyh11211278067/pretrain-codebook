import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


def conv3x3(inplanes, outplanes, stride=1):
    """
    一个简单的3x3卷积填充包装。
    参数:
        inplanes (int):输入的通道数。
        outplanes (int):输出的通道数。
        stride (int):卷积的步幅。默认值:1。
    """
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):  # ResNet架构中的基本组件
    """
    在ResNetArcFace架构中使用的基本剩余块。
    参数:
        inplanes (int):输入的通道数。
        planes (int):输出的通道数。
        stride (int):卷积的步幅。默认值:1。
        downsample (nn.Module): downsample模块。默认值:没有。
    """
    expansion = 1  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    # 在 forward 方法中，首先保存了输入的原始数据 x，作为残差连接的一部分。
    # 然后，数据通过第一个卷积层、批量归一化层和ReLU激活函数进行处理。
    # 接着，处理后的数据通过第二个卷积层和批量归一化层。
    # 最后，将经过两个卷积层处理后的数据与残差连接的数据相加，并再次通过ReLU激活函数，得到输出。
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


# 用作改进的残差块
# SEBlock通过两个操作——挤压（squeeze）和激励（excitation）来重新标定特征图。
# 挤压操作通过全局平均池化来压缩特征图的空间维度，从而得到每个通道的描述符。
# 激励操作则通过一个简单的门控机制来学习每个通道的重要性，并为每个通道生成一个权重。
# 这些权重随后被用来对原始特征图进行重标定。
class IRBlock(nn.Module):
    """
    改进的残差块(IR block)在ResNetArcFace架构中使用。
    参数:
        inplanes (int):输入的通道数。
        planes (int):输出的通道数。
        stride (int):卷积的步幅。默认值:1。
        downsample (nn.Module): downsample模块。默认值:没有。
        use_se (bool):是否使用SEBlock(挤压和激励块)。默认值:真的。
    """
    expansion = 1  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


# Bottleneck块采用了“瓶颈”设计，即先通过1x1卷积减少通道数，然后通过一个3x3卷积进行特征提取，最后再通过一个1x1卷积恢复通道数。这种设计可以减少计算量，同时保持网络的性能。
class Bottleneck(nn.Module):
    """
    瓶颈块在ResNetArcFace架构中使用。
    参数:
        inplanes (int):输入的通道数。
        planes (int):输出的通道数。
        stride (int):卷积的步幅。默认值:1。
        downsample (nn.Module): downsample模块。默认值:没有。
    """
    expansion = 4  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # 首先保存了输入的原始数据x作为残差连接的一部分。
    # 然后，数据通过第一个卷积层（conv1）、第一个批量归一化层（bn1）和ReLU激活函数进行处理。
    # 接着，数据通过第二个卷积层（conv2）、第二个批量归一化层（bn2）和ReLU激活函数进行进一步处理。
    # 最后，数据通过第三个卷积层（conv3）和第三个批量归一化层（bn3）。
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """
    在IRBlock中使用的挤压和激励块(SEBlock)。
    参数:
        channel (int):输入的通道数。
        reduction (int):通道缩减比例。默认值:16。
    """

    # 在初始化方法中，定义了一个自适应平均池化层（avg_pool），它将输入特征图的空间维度压缩为1x1，从而得到每个通道的全局信息。
    # 接着定义了一个全连接层序列（fc），它包含两个全连接层，中间夹着PReLU激活函数。
    # 第一个全连接层将通道数从channel减少到channel // reduction（即缩减了reduction倍），第二个全连接层则恢复通道数至channel。
    # 最后，通过Sigmoid激活函数将输出值限制在0到1之间，得到每个通道的权重。
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # pool to 1x1 without spatial information
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.PReLU(), nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    # 在forward方法中，首先获取输入张量x的批量大小（b）和通道数（c）。
    # 然后，通过自适应平均池化层对输入特征图进行空间维度的压缩，得到每个通道的全局信息。
    # 接下来，将压缩后的特征图展平（view），并通过全连接层序列进行处理，得到每个通道的权重。
    # 最后，将这些权重重新塑形为与原始特征图相同的空间尺寸（1x1），并与原始特征图进行逐元素相乘，从而得到SE块的输出。
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 通过堆叠多个自定义的block类型的块，以及使用批量归一化、激活函数和池化层等操作，模型能够提取出有效的特征表示。
@ARCH_REGISTRY.register()
class ResNetArcFace(nn.Module):
    """
    ArcFace与ResNet架构。
    参考:ArcFace:深度人脸识别的加性角边缘损失。
    参数:
        block (str):在ArcFace架构中使用的块。
        layers (tuple(int)):每层中的块数。
        use_se (bool):是否使用SEBlock(挤压和激励块)。默认值:真的。
    """

    def __init__(self, block, layers, use_se=True):
        if block == 'IRBlock':
            block = IRBlock
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetArcFace, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1,
                               bias=False)  # 第一个卷积层，输入通道数为1（通常用于灰度图像），输出通道数为64，卷积核大小为3x3，步长为1，填充为1，不使用偏置。
        self.bn1 = nn.BatchNorm2d(64)  # 第一个批量归一化层。
        self.prelu = nn.PReLU()  # PReLU激活函数层。
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，核大小为2x2，步长为2。

        # 然后，通过_make_layer方法（该方法在这段代码中未给出）构建了四个层次的块，分别对应layer1、layer2、layer3和layer4。
        # 每个层次中的块数量和特征图的通道数由layers参数和内部定义的self.inplanes变量决定。
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn4 = nn.BatchNorm2d(512)  # 第四个层次之后的批量归一化层。
        self.dropout = nn.Dropout()   # Dropout层，用于防止过拟合。
        self.fc5 = nn.Linear(512 * 8 * 8, 512)  # 全连接层，将特征图展平后的特征转换为512维的向量。
        self.bn5 = nn.BatchNorm1d(512)  # 全连接层之后的批量归一化层。

        # initialization 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # _make_layer方法负责构建网络中的一个层次，这个层次由多个block组成。
    # 这个方法首先检查是否需要下采样，如果需要，则创建一个包含卷积和批量归一化的下采样序列。
    # 然后，它创建一个block的列表，其中第一个block可能包含下采样序列，并且具有特定的步长。
    # 其余的block则没有下采样，并且步长为1。最后，这个方法返回由这些block组成的序列。
    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    # forward方法定义了模型的前向传播过程。输入数据首先经过一个初始的卷积层、批量归一化层、激活函数和最大池化层。
    # 然后，数据依次通过四个由_make_layer方法构建的网络层次。在每个层次之后，可能会进行批量归一化和Dropout操作。
    # 最后，数据被展平并通过一个全连接层，可能再次进行批量归一化。
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return x

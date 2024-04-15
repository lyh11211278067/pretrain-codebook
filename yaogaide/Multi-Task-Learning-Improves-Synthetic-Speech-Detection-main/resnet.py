import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np


# 这个SelfAttention类实现了一个简单的自注意力机制，它使用可学习的权重对输入进行加权，并可以选择返回加权输入的均值或均值与标准差的组合。
# 定义一个名为SelfAttention的类，它继承了PyTorch的nn.Module基类，用于创建自定义的神经网络模块。
class SelfAttention(nn.Module):
    # 初始化函数，接受两个参数：hidden_size（隐藏层的大小）和mean_only（一个布尔值，决定是否只返回加权输入的均值）。
    def __init__(self, hidden_size, mean_only=False):
        # 调用父类nn.Module的初始化函数。
        super(SelfAttention, self).__init__()

        # self.output_size = output_size
        # 保存传入的hidden_size作为类的属性。
        self.hidden_size = hidden_size
        # 定义一个名为att_weights的参数，它是一个形状为(1, hidden_size)的张量，并设置其requires_grad属性为True，使其可以在训练过程中被优化。
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        # 保存传入的mean_only作为类的属性。
        self.mean_only = mean_only
        # 使用Kaiming均匀初始化方法初始化att_weights。
        init.kaiming_uniform_(self.att_weights)

    # 定义前向传播函数，接受一个名为inputs的参数。
    def forward(self, inputs):
        # 获取输入inputs的批处理大小。
        batch_size = inputs.size(0)
        # 计算输入inputs和att_weights之间的批量矩阵乘法。首先，对att_weights进行变换，使其形状与inputs匹配，然后执行批量矩阵乘法。
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        # 检查输入inputs的批处理大小是否为1。
        # 如果批处理大小为1，则对weights应用tanh激活函数，然后在其上执行softmax操作，得到注意力权重。
        # 使用注意力权重对输入inputs进行加权。
        if inputs.size(0) == 1:
            attentions = F.softmax(torch.tanh(weights), dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        # 如果批处理大小不为1：
        # 对weights进行squeeze操作，然后应用tanh激活函数，并在其上执行softmax操作，得到注意力权重。
        # 使用注意力权重对输入inputs进行加权。
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()), dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        # 如果mean_only为True：只返回加权输入的均值。
        if self.mean_only:
            return weighted.sum(1)
        # 如果mean_only不为True：生成与weighted相同形状的随机噪声。
        else:
            noise = 1e-5 * torch.randn(weighted.size())
            # 如果输入inputs在CUDA设备上，则将噪声也移动到相同的设备上。
            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            # 计算加权输入的均值和标准差。
            avg_repr, std_repr = weighted.sum(1), (weighted + noise).std(1)
            # 将均值和标准差拼接在一起，形成新的表示。
            representations = torch.cat((avg_repr, std_repr), 1)
            # 返回新的表示。
            return representations


# 这是一个名为PreActBlock的类，它继承自PyTorch的nn.Module。这是一个神经网络模块，表示一个预激活的基本块（Pre-activation BasicBlock）。
# 这个PreActBlock类实现了一个预激活的残差块
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    # 设置类的静态属性expansion，值为1。这个属性通常用于决定块的输出通道数与输入通道数之间的比例。
    expansion = 1

    # 类的构造函数，用于初始化对象
    # in_planes: 输入的通道数。
    # planes: 输出的通道数。
    # stride: 卷积操作的步长。
    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        # super(PreActBlock, self).__init__()调用了父类nn.Module的初始化方法。
        super(PreActBlock, self).__init__()
        # 接下来，定义了两个批归一化层（BatchNorm2d）和两个卷积层（Conv2d）。
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # 接着，有一个条件语句来判断是否需要添加一个快捷路径（shortcut）。当步长不为1或输入通道数与输出的通道数不匹配时，需要添加这个快捷路径。
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    # 接下来是forward方法，它定义了数据通过该块的前向传播路径。x: 输入数据。
    def forward(self, x):
        # 首先对输入数据进行批归一化并使用ReLU激活函数。
        out = F.relu(self.bn1(x))
        # 接着，检查是否存在快捷路径。如果存在，就应用快捷路径；否则，直接使用输入x作为快捷路径。
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # 对out进行第一个卷积操作。
        out = self.conv1(out)
        # 对out进行批归一化，然后使用ReLU激活函数，再进行第二个卷积操作。
        out = self.conv2(F.relu(self.bn2(out)))
        # 最后，将out与快捷路径shortcut相加，完成残差连接。
        out += shortcut
        # 返回结果。
        return out


# 这个PreActBottleneck类是一个预激活的瓶颈（Bottleneck）模块，它是深度神经网络中常用的结构，特别是在ResNet等模型中。
# 这个模块与PreActBlock类似，但具有更多的卷积层和不同的通道数配置，以形成瓶颈形状。
class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module.它是原始Bottleneck模块的预激活版本。"""
    # 设置类的静态属性expansion，值为4。这表示Bottleneck模块的输出通道数将是输入通道数的4倍。
    expansion = 4

    # 定义类的构造函数__init__，它接受输入通道数in_planes、中间通道数planes、步长stride以及额外的参数。
    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        # 调用父类nn.Module的初始化方法。
        super(PreActBottleneck, self).__init__()
        # self.bn1和self.conv1构成第一个1x1卷积层，用于减少通道数。
        # self.bn2和self.conv2构成3x3卷积层，是主要的特征提取层。
        # self.bn3和self.conv3构成第二个1x1卷积层，用于增加通道数，恢复到expansion倍的输入通道数。
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        # 接下来的代码检查是否需要添加快捷路径（shortcut）：
        # 如果步长不等于1，或者输入通道数不等于输出通道数的expansion倍，则添加一个1x1的卷积作为快捷路径，以匹配维度。
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    # 定义前向传播方法forward，接受输入x。
    def forward(self, x):
        # 首先，对输入x进行批归一化（self.bn1）并应用ReLU激活函数。
        # 然后，检查是否存在快捷路径。如果存在，则应用快捷路径；否则，直接使用输入x作为快捷路径。
        # 接着，依次通过第一个1x1卷积、3x3卷积和第二个1x1卷积，每次卷积后都进行批归一化和ReLU激活（除了最后一个卷积后没有激活）。
        # 最后，将输出out与快捷路径相加，完成残差连接。
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


# 这个函数返回一个3x3的卷积层，其中in_planes是输入通道数，out_planes是输出通道数，stride是卷积步长。
# 该卷积层使用了1的填充（padding=1），这意味着输入和输出的宽度和高度将保持不变（当步长为1时）。
# 此外，该卷积层不使用偏置项（bias=False）。
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# 这个函数返回一个1x1的卷积层，参数与conv3x3类似。1x1的卷积层常用于改变通道数，而不改变特征图的空间维度。
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 这个字典包含了不同ResNet模型（如ResNet-18, ResNet-50等）的配置。每个模型的配置是一个包含四个元素的列表，表示每个阶段（通常是一个或多个卷积块）的块数。
# 例如，在ResNet-50中，第一个阶段有3个块，第二个阶段有4个块，依此类推。与这些配置一起，还指定了每个阶段使用的块类型（PreActBlock或PreActBottleneck）。
RESNET_CONFIGS = {
    'recon': [[1, 1, 1, 1], PreActBlock],
    '18': [[2, 2, 2, 2], PreActBlock],
    '28': [[3, 4, 6, 3], PreActBlock],
    '34': [[3, 4, 6, 3], PreActBlock],
    '50': [[3, 4, 6, 3], PreActBottleneck],
    '101': [[3, 4, 23, 3], PreActBottleneck]
}


# 这个函数用于设置各种库的随机种子，以确保实验的可重复性。它设置了PyTorch、Python标准库（random）、NumPy和CuDNN的随机种子。
# 如果CUDA可用，它还会设置CUDA的随机种子，并确保CuDNN的确定性模式（deterministic=True），这意味着每次给定相同的输入时，卷积操作都会返回相同的结果。
# benchmark=False表示禁用CuDNN的自动调优功能，以确保确定性。
def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False


# 定义了一个名为ResNet的类，它继承了PyTorch的nn.Module基类，表示这是一个神经网络模型。
class ResNet(nn.Module):
    # num_nodes: 自定义卷积层conv5中的节点数。
    # enc_dim: 全连接层fc的输出维度。
    # resnet_type: 指定ResNet的类型，例如'18'、'50'等。
    # nclasses: 输出类别的数量。
    # dropout1d和dropout2d: 是否使用1D和2D的dropout。
    # p: dropout的概率。
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2, dropout1d=False, dropout2d=False, p=0.01):
        # 设置初始通道数为16。
        self.in_planes = 16
        # 调用父类nn.Module的初始化方法。
        super(ResNet, self).__init__()
        # 根据resnet_type从RESNET_CONFIGS字典中获取对应的层配置和块类型。
        layers, block = RESNET_CONFIGS[resnet_type]
        # 设置批归一化层的类型为nn.BatchNorm2d。
        self._norm_layer = nn.BatchNorm2d
        # 定义一个二维卷积层conv1，输入通道数为1，输出通道数为16，卷积核大小为(9, 3)，步长为(3, 1)，填充为(1, 1)，不使用偏置项。
        # 定义一个批归一化层bn1，用于conv1的输出。
        # 定义一个ReLU激活函数activation。
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()
        # 构建了ResNet的主要部分（layer1, layer2, layer3, layer4），这些层由不同的残差块组成，通过_make_layer函数来构建。
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 设置是否使用1D和2D的dropout的布尔值。
        self.if_dropout1d = dropout1d
        self.if_dropout2d = dropout2d
        # 如果dropout2d为True，则创建一个二维的dropout层。
        if self.if_dropout2d:
            self.dropout2d = nn.Dropout2d(p=p, inplace=True)
        # layer3, layer4），这些层由不同的残差块组成，通过_make_layer函数来构建。
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 定义一个特殊的二维卷积层conv5，其输入通道数取决于残差块的扩展倍数（block.expansion），
        # 输出通道数为256，卷积核大小为(num_nodes, 3)，步长和填充分别为(1, 1)和(0, 1)。
        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        # 定义一个批归一化层bn5，用于conv5的输出。
        self.bn5 = nn.BatchNorm2d(256)
        # 如果dropout1d为True，则创建一个一维的dropout层。
        if self.if_dropout1d:
            self.dropout1d = nn.Dropout(p=p, inplace=True)
        # 定义一个全连接层fc，输入维度为512（256 * 2），输出维度为enc_dim。
        self.fc = nn.Linear(256 * 2, enc_dim)
        # 这行代码定义了另一个全连接层fc_logvar，它的输入维度也是enc_dim，输出维度取决于nclasses。
        # 如果nclasses大于或等于2，则输出维度为nclasses；否则，输出维度为1。
        # 这通常用于变分自编码器（Variational Autoencoder, VAE）中，其中fc_mu和fc_logvar分别用于输出均值和对数方差。
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        # initialize_params方法，用于初始化模型的参数。
        self.initialize_params()
        # 实例化为self.attention，并传入参数256
        self.attention = SelfAttention(256)
        # 这行代码创建了一个GRU（Gated Recurrent Unit）循环神经网络层，并将其实例化为self.GRU。nn.GRU是PyTorch中提供的GRU实现。这个GRU层有三个参数：
        #
        # 第一个参数94是输入特征的数量，即每个时间步的输入向量的维度。
        # 第二个参数94是隐藏层的维度，即GRU单元内部的隐藏状态的维度。这里输入和输出的维度相同，意味着GRU不进行任何特征变换，只是基于输入和之前的隐藏状态计算新的隐藏状态。
        # 第三个参数5是GRU层的层数，即堆叠的GRU单元数量。这意味着该层包含5个堆叠的GRU单元，每个单元都会接收前一个单元的输出作为输入。
        self.GRU = nn.GRU(94, 94, 5)

    """
    这个函数用于初始化模型中的权重。它遍历了模型中的所有层，并根据层的类型应用不同的初始化方法：
    对于torch.nn.Conv2d（二维卷积层），使用kaiming_normal_初始化方法，这是针对ReLU激活函数的权重初始化方法。
    对于torch.nn.Linear（全连接层），使用kaiming_uniform_初始化方法，这也是针对ReLU激活函数的权重初始化方法。
    对于torch.nn.BatchNorm2d和torch.nn.BatchNorm1d（批量归一化层），将权重设置为1，偏置设置为0。
    """

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    """
    这个函数用于构建网络中的层，特别是残差块。
    它接受一个块类型（如BasicBlock或Bottleneck），输出通道数planes，块的数量num_blocks，以及步长stride作为参数。
    函数返回一个由多个相同类型的块组成的序列。
    """

    def _make_layer(self, block, planes, num_blocks, stride=1):
        # 初始化下采样（如果需要的话）
        norm_layer = self._norm_layer
        # 初始化downsample为None，它将用于下采样输入特征图以匹配残差块的维度。
        downsample = None
        # 判断是否需要下采样。如果需要下采样，创建一个下采样序列，包括一个卷积层（用于改变通道数）和一个批量归一化层。
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        # 创建残差块层
        # 初始化一个空列表layers，用于存储要构建的残差块。
        layers = []
        # 向layers列表中添加第一个残差块，传入当前输入通道数self.inplanes，输出通道数planes，步长stride，以及可能的下采样序列downsample。
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        # 更新self.inplanes，它是下一个残差块的输入通道数。block.expansion是块中通道数的扩展因子（通常是4）。
        self.in_planes = planes * block.expansion
        # 循环blocks-1次，以添加剩余的残差块。
        for _ in range(1, num_blocks):
            # 在循环中，向layers列表中添加更多的残差块，这些块具有与第一个块相同的输出通道数和步长。
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        # 返回一个包含所有残差块的nn.Sequential容器。使用*layers将列表解包为单独的参数。
        return nn.Sequential(*layers)

    """
    前向传播函数定义了输入数据x通过网络的方式。这里简要概述了剩余的步骤：
    1、数据通过一系列的卷积层、批量归一化层、激活函数以及残差块（由_make_layer函数构建）。
    2、数据经过self.conv5卷积层后，通过批归一化和激活函数，并通过squeeze(2)操作移除了大小为1的维度（通常用于移除单通道维度）。
    3、处理后的数据x经过自注意力层self.attention，其中x.permute(0, 2, 1).contiguous()是为了调整数据的维度顺序以适应自注意力层的要求。
    4、自注意力层的输出stats被传递给一个全连接层self.fc，生成特征feat。
    5、特征feat再传递给另一个全连接层self.fc_mu，生成均值mu。
    6、函数返回特征feat和均值mu，这通常用于变分自编码器（VAE）中，其中mu是潜在空间的均值。
    这个模型似乎是一个结合了残差网络（ResNet）和自注意力机制的变分自编码器。通过自注意力机制，模型能够捕捉输入数据中的长距离依赖关系，而残差连接则有助于模型训练时的梯度传播。
    """

    def forward(self, x):
        # 输入x首先经过self.conv1卷积层
        x = self.conv1(x)
        # 然后通过self.bn1批量归一化层
        x = self.activation(self.bn1(x))
        # 接下来，x通过self.layer的四组残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 通过self.conv5卷积层
        x = self.conv5(x)
        # 通过激活函数（很可能是ReLU）
        x = self.activation(self.bn5(x)).squeeze(2)

        # 将x的维度重新排列以适应自注意力层的要求
        # 通常，自注意力层期望的输入是[batch_size, channels, height, width]
        # 但是前面的网络可能输出的是[batch_size, height, width, channels]
        # permute操作将维度重新排列为[batch_size, channels, height, width]
        stats = self.attention(x.permute(0, 2, 1).contiguous())
        # 将经过自注意力层处理后的特征图stats传递给全连接层self.fc
        # 生成的特征可能用于进一步的处理，如分类或解码等
        feat = self.fc(stats)
        # 将特征feat传递给另一个全连接层self.fc_mu
        # 这通常是在变分自编码器（VAE）中，用于生成潜在空间的均值（mu）
        mu = self.fc_mu(feat)
        # 函数返回特征feat和均值mu
        return feat, mu


# 结合了卷积神经网络（CNN）和全连接层的自编码器（autoencoder）结构，用于数据重构。
# enc_dim: 输入特征的维度。
# resnet_type: 指定ResNet的类型。
# nclasses: 类别数量。
"""
self.fc: 一个全连接层，将enc_dim维度的输入转化为4 * 10 * 125维度的输出。
self.bn1: 一个2D批量归一化层，用于归一化具有4个通道的特征图。
self.activation: ReLU激活函数。
self.layer1, self.layer2, self.layer3, self.layer4: 这些是包含多个PreActBlock（预激活块）的序列模型。每个PreActBlock都包含批量归一化、ReLU激活和卷积操作。
"""


class Reconstruction_autoencoder(nn.Module):
    def __init__(self, enc_dim, resnet_type='18', nclasses=2):
        super(Reconstruction_autoencoder, self).__init__()

        self.fc = nn.Linear(enc_dim, 4 * 10 * 125)
        self.bn1 = nn.BatchNorm2d(4)
        self.activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            PreActBlock(4, 16, 1),
            PreActBlock(16, 64, 1),
            PreActBlock(64, 128, 1)
        )

        self.layer2 = nn.Sequential(
            PreActBlock(128, 64, 1)
        )
        self.layer3 = nn.Sequential(
            PreActBlock(64, 32, 1),
            PreActBlock(32, 16, 1)
        )
        self.layer4 = nn.Sequential(
            PreActBlock(16, 8, 1),
            PreActBlock(8, 4, 1),
            PreActBlock(4, 1, 1),
        )

    """
    z = self.fc(z).view((z.shape[0], 4, 10, 125)): 首先，通过全连接层self.fc改变z的形状，并重新排列其维度。
    z = self.activation(self.bn1(z)): 使用批量归一化层self.bn1对z进行归一化，并通过ReLU激活函数。
    z = self.layer1(z), z = self.layer2(z), z = self.layer3(z), z = self.layer4(z): z依次通过四个预激活块序列。
    z = nn.functional.interpolate(z, scale_factor=3, mode="bilinear", align_corners=True): 在self.layer2之后，使用双线性插值将z的尺寸放大3倍。
    z = nn.functional.interpolate(z, scale_factor=2, mode="bilinear", align_corners=True): 在self.layer3之后，使用双线性插值将z的尺寸放大2倍。
    """

    def forward(self, z):
        z = self.fc(z).view((z.shape[0], 4, 10, 125))
        z = self.activation(self.bn1(z))
        z = self.layer1(z)
        z = self.layer2(z)
        z = nn.functional.interpolate(z, scale_factor=3, mode="bilinear", align_corners=True)
        z = self.layer3(z)
        z = nn.functional.interpolate(z, scale_factor=2, mode="bilinear", align_corners=True)
        z = self.layer4(z)
        return z


class compress_Block(nn.Module):
    expansion = 1

    # in_planes: 输入特征图的通道数。
    # planes: 输出特征图的通道数。
    # stride: 卷积操作的步长。
    # *args, **kwargs: 其他可选参数，但在这个类的定义中并未使用。
    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(compress_Block, self).__init__()

        # self.bn1: 批量归一化层，用于对输入特征图进行归一化。
        # self.conv1: 第一个卷积层，将输入特征图从in_planes通道数转换到planes通道数。
        # self.bn2: 另一个批量归一化层，用于对self.conv1的输出进行归一化。
        # self.conv2: 第二个卷积层，保持特征图的通道数为planes。
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # 如果stride不等于1，或者输入通道数in_planes不等于self.expansion * planes（其中self.expansion被硬编码为1），则定义一个快捷方式（shortcut）。
        # 这个快捷方式是一个1x1的卷积层，用于调整输入特征图的通道数或尺寸，使其与主路径的输出相匹配。
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        # 首先，输入特征图x通过批量归一化层self.bn1，然后应用ReLU激活函数。
        out = F.relu(self.bn1(x))
        # 判断是否存在快捷方式。如果存在，则将out通过快捷方式；如果不存在，则直接使用out作为快捷方式的结果。
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # 将out通过第一个卷积层self.conv1。
        out = self.conv1(out)
        # 对out进行批量归一化、ReLU激活，然后通过第二个卷积层self.conv2。
        out = self.conv2(F.relu(self.bn2(out)))
        # 将主路径的输出与快捷方式的结果相加，实现残差连接。
        out += shortcut
        return out

# BasicBlock的预激活版本，同上
class compress_block(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(compress_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

# 这个类实现了一个自编码器（Autoencoder）的结构，自编码器通常用于无监督学习，目的是学习输入数据的压缩和编码表示。
# 这个特定的自编码器结构似乎是基于ResNet（残差网络）的变种，并使用了自定义的compress_block和PreActBlock（预激活块）作为其基本构建块。
class Conversion_autoencoder(nn.Module):
    def __init__(self, num_nodes, enc_dim, nclasses=2):
        # num_nodes, enc_dim, 和 nclasses 是传递给类的参数。
        # self.in_planes 被初始化为16，表示输入特征图的初始通道数。
        # layers 和 block 是从RESNET_CONFIGS['recon']中获取的。
        # self._norm_layer 被设置为nn.BatchNorm2d，表示将使用批量归一化层。
        self.in_planes = 16
        super(Conversion_autoencoder, self).__init__()

        layers, block = RESNET_CONFIGS['recon']

        self._norm_layer = nn.BatchNorm2d

        # self.conv1 和 self.bn1 定义了第一个卷积层和批量归一化层。
        # self.layer1, self.layer2, self.layer3, 和 self.layer4 是编码器的不同层，它们由compress_block组成。
        # 每个层都包含两个compress_block，并且随着层的深入，特征图的通道数逐渐增加，同时尺寸减小（通过步长为2的卷积实现）。
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            compress_block(16, 16, 1),
            compress_block(16, 32, (2, 3)),
        )

        self.layer2 = nn.Sequential(
            compress_block(32, 32, 1),
            compress_block(32, 64, 2),
        )

        self.layer3 = nn.Sequential(
            compress_block(64, 64, 1),
            compress_block(64, 128, 2),
        )

        self.layer4 = nn.Sequential(
            compress_block(128, 256, 1),
            compress_block(256, 128, 1),
        )

        # connect x_1
        # self.layer1_i, self.layer2_i, self.layer3_i, 和 self.layer4_i 是解码器的不同层。解码器的目的是从编码器的输出中重建原始输入。
        # 每层都使用nn.ConvTranspose2d进行上采样（即增加特征图的尺寸），并使用PreActBlock进行进一步的处理。
        # 注意到解码器的结构与编码器是对称的，但不是完全相同的。
        # 例如，编码器中的self.layer1有两个compress_block，而解码器中的self.layer1_i有三个层（一个上采样卷积和两个PreActBlock）。
        self.layer1_i = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, 2, 1),
            PreActBlock(256, 128, 1),
            PreActBlock(128, 64, 1),
        )

        self.layer2_i = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            PreActBlock(128, 64, 1),
            PreActBlock(64, 32, 1),
        )
        self.layer3_i = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, (2, 3), 1, output_padding=(1, 2)),
            PreActBlock(64, 32, 1),
            PreActBlock(32, 16, 1)
        )
        self.layer4_i = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (9, 3), (3, 2), 1, output_padding=(2, 1)),
            PreActBlock(32, 8, 1),
            PreActBlock(8, 4, 1),
            PreActBlock(4, 1, 1),
        )

    # 这个方法的主要目的是初始化网络中的参数。它遍历网络中的所有层，然后根据层的类型进行不同的初始化。
    # 对于 torch.nn.Conv2d（二维卷积层）类型的层，它使用 kaiming_normal_ 方法进行权重初始化，这是一种专门为ReLU激活函数设计的初始化方法。
    # 对于 torch.nn.Linear（全连接层）类型的层，它使用 kaiming_uniform_ 方法进行权重初始化。
    # 对于 torch.nn.BatchNorm2d 和 torch.nn.BatchNorm1d（批量归一化层）类型的层，它将权重设置为1，偏置设置为0。
    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    # 这个方法用于构建一个层的序列，通常是ResNet中的某个阶段（例如，conv2_x、conv3_x 等）。
    #
    # 它首先检查是否需要下采样（即改变输入数据的尺寸或通道数）。如果需要，它会创建一个下采样序列，该序列包含一个1x1的卷积层（用于改变通道数）和一个批量归一化层。
    # 然后，它创建一个新的层，该层是该阶段的第一层（即一个残差块）。这个层的步长可能与1不同，具体取决于是否需要下采样。
    # 接下来，它更新 self.in_planes 的值，这个值通常表示输入到下一个残差块的特征图的通道数。
    # 最后，它创建并添加剩余的残差块到层序列中。这些块的步长为1，不需要改变输入数据的尺寸或通道数。
    # 这个方法最后返回一个包含所有这些层的序列，这些层通常会被添加到网络的其他部分。
    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        # 输入x首先经过一个卷积层conv1，然后通过激活函数和批量归一化。
        x = self.conv1(x)
        x_1 = self.activation(self.bn1(x))
        # 接下来，x经过四个残差块序列layer1、layer2、layer3和layer4，每个序列都包含多个残差单元。
        x_2 = self.layer1(x_1)
        x_3 = self.layer2(x_2)
        x_4 = self.layer3(x_3)
        x_5 = self.layer4(x_4)
        # 在每个残差块之后，模型通过反向连接（即短路连接）将前一层的输出与当前层的输出拼接起来，并传递给下一个处理层。
        # 这种结构有助于网络学习恒等映射，从而减轻梯度消失或表示瓶颈问题。
        # torch.cat是将两个参数拼接，self.layer则是将拼接后的y_传递给另一个处理层
        y_1 = torch.cat([x_5, x_4], dim=1)
        y_2 = self.layer1_i(y_1)
        y_2 = torch.cat([y_2, x_3], dim=1)
        y_3 = self.layer2_i(y_2)
        y_3 = torch.cat([y_3, x_2], dim=1)
        y_4 = self.layer3_i(y_3)
        y_5 = torch.cat([y_4, x_1], dim=1)
        result = self.layer4_i(y_5)
        return result

# 一个简单的全连接网络，用于分类任务，特别是针对说话者识别
# 该网络由三个全连接层（fc_1、fc_2、fc_3）和两个批量归一化层（bn_1、bn_2）组成。每个全连接层后面都跟着一个ReLU激活函数，除了最后一个全连接层外。
# 这个网络将enc_dim维的输入向量转换为nclasses维的输出向量，其中nclasses是说话者分类任务中的类别数。输出向量可以直接用于分类，如通过softmax函数得到每个类别的概率。
class Speaker_classifier(nn.Module):
    def __init__(self, enc_dim, nclasses):
        super(Speaker_classifier, self).__init__()
        # 第一个全连接层，将enc_dim维的输入转换为128维的输出。
        self.fc_1 = nn.Linear(enc_dim, 128)
        # 对128维的输出进行批量归一化。
        self.bn_1 = nn.BatchNorm1d(128)
        # 第二个全连接层，将128维的输入转换为64维的输出。
        self.fc_2 = nn.Linear(128, 64)
        # 对64维的输出进行批量归一化。
        self.bn_2 = nn.BatchNorm1d(64)
        # 第三个全连接层，将64维的输入转换为nclasses维的输出，即分类的类别数。
        self.fc_3 = nn.Linear(64, nclasses)

    def forward(self, x):
        # self.fc_1(x): 将输入x传递给第一个全连接层fc_1。这个全连接层会对输入数据进行线性变换，即计算输入数据与权重的点积，并加上偏置项。
        # self.bn_1(...): 将全连接层的输出传递给第一个批量归一化层bn_1。批量归一化会对数据进行标准化，即减去均值并除以标准差，从而使数据具有零均值和单位方差。
        # 这有助于加速训练和提高模型的稳定性。
        # F.relu(...): 将批量归一化的输出传递给ReLU激活函数。ReLU函数会将所有负值置为零，保留非负值。这样做可以引入非线性因素，使得网络能够学习更复杂的模式。
        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        # 这行代码将经过两个全连接层和两个批量归一化层处理后的数据x传递给第三个全连接层fc_3。这个全连接层会进行最后一轮的线性变换，生成模型的输出y。
        # 注意，这一层的输出通常不经过激活函数，因为输出层的激活函数通常是在模型外部的softmax函数（对于分类任务），或者是在损失函数中隐式地应用的。
        y = self.fc_3(x)
        return y

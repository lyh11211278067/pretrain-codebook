import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, pos_weight=1.0, neg_weight=1.0, beta=2.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')
        self.softplus = nn.Softplus()

    def forward(self, pred, target, labels):
        # 计算基础 MSE 损失，保留逐元素损失
        loss = self.mse(pred, target)

        # 计算每个样本的损失
        loss = loss.view(loss.size(0), -1).mean(dim=1)

        # 创建正样本和负样本的掩码
        pos_mask = (labels == 1).float()
        neg_mask = (labels == 0).float()

        # 对正样本加权，以增强其影响
        weighted_loss = self.beta * self.pos_weight * pos_mask * loss - self.neg_weight * neg_mask * loss

        # 应用 Softplus 函数
        softplus_loss = self.softplus(self.alpha * weighted_loss)

        # 返回最终的损失值
        return softplus_loss.mean()


# 示例用法
batch_size = 8
channels = 3
height = 64
width = 64

# 随机生成预测值和目标值
pred = torch.randn(batch_size, channels, height, width)
target = torch.randn(batch_size, channels, height, width)
labels = torch.randint(0, 2, (batch_size,))

# 实例化自定义损失函数
criterion = CustomLoss(alpha=20.0, pos_weight=1.0, neg_weight=1.0, beta=2.0)

# 计算自定义损失
loss = criterion(pred, target, labels)
print('Custom Loss:', loss.item())

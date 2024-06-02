import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, -output_scores.squeeze(1)

class AMSoftmax(nn.Module):
    def __init__(self, num_classes, enc_dim, s=20, m=0.9):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits


def MSELoss(recons, input, vq_loss,weight=0.5):
    recons_loss = F.mse_loss(recons, input)
    loss = recons_loss + vq_loss * weight

    return loss, recons_loss, vq_loss


class CodeWeightLoss(nn.Module):
    def __init__(self, alpha=1.0, pos_weight=1.0, neg_weight=1.0, beta=2.0):
        super(CodeWeightLoss, self).__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')
        self.softplus = nn.Softplus()

    def forward(self, pred, gt, labels):
        # 计算基础 MSE 损失
        loss = self.mse(pred, gt)
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
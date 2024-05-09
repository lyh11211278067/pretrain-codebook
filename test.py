import torch
import torch.nn.functional as F

# 创建大小为 (60, 704) 的随机数据作为示例输入
data = torch.randn(60, 704)

# 定义需要填充的维度和填充值
pad_dims = (0, 64 - 60)  # 在第一个维度（行）上填充 4 行
pad_value = 0  # 填充值为 0

# 进行填充操作
padded_data = F.pad(data, pad=(0, 0, 0, 4), mode='constant', value=pad_value)

# 检查填充后的数据尺寸
print("填充后的数据尺寸：", padded_data.shape)
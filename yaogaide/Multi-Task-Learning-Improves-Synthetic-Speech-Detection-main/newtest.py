import torch

print(torch.__version__)  # 打印PyTorch版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.cuda.get_device_name(0))  # 打印第一个CUDA设备的名称

import torch   #查看torch版本
print(torch.__version__)  #注意是双下划线

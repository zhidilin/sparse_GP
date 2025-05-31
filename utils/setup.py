import numpy as np
import torch
import math

def reset_seed(seed: int) -> None:
    # 生成随机数，以便固定后续随机数，方便复现代码
    np.random.seed(seed)
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(seed)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(seed)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(seed)

# torch_version = '1.8.1'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
## Constant definitions
pi = torch.tensor(math.pi).to(device)

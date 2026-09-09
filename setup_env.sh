ran#!/bin/zsh
# 为 conda env `anomaly` (Python 3.10) 安装 SECOND 项目所需依赖
# 用法: bash setup_env.sh

PIP=/home/yiruolei/anaconda3/envs/anomaly/bin/pip

# 1. PyTorch (CUDA 12.4 wheel, 兼容 driver 570 / CUDA 12.8, 支持 RTX 5880 Ada sm_89)
$PIP install torch --index-url https://download.pytorch.org/whl/cu124

# 2. 其余依赖(代码实际 import 的全部第三方包)
#    requirments 里的 zhusuan/tfsnentence 是旧 Donut 代码遗留, 现在 import 不到, 不装
$PIP install numpy scipy scikit-learn pandas tqdm matplotlib einops scienceplots

# 3. 验证
/home/yiruolei/anaconda3/envs/anomaly/bin/python - <<'EOF'
import torch, numpy, scipy, sklearn, pandas, tqdm, matplotlib, einops, scienceplots
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("all imports OK")
EOF

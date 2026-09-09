#!/bin/zsh 
eval "$(conda shell.bash hook)"
conda activate loranomaly 
export Name='new结构不是lstm试一试'
# export LOAD='/home/yiruolei/project/anomaly-detection/SecondTask/checkpoint/onlycontrast/SMAP/best.pt'
# export LOAD='/home/yiruolei/project/anomaly-detection/SecondTask/checkpoint/onlycontrast/SMAP/9.pt'
python main.py --name $Name  --train  
#  python main.py --name $Name --loadpath $LOAD 
# >> $Name.out 2>&1 &
#貌似因为上面的代码没有和out这个要分开写？怎么整体使用？？

#todo 目前是没有使用对比loss的，只是加了框架，这个loss加入后的效果明天是
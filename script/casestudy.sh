#这个的出现完全是因为pth保存问题
#!/bin/zsh 
# eval "$(conda activate loranomaly  hook)"
# export Name='casestudy'
# export data='MSL'
export data='WADI'
export MODE='train-test'
export LOAD=None
# export CUDA_VISIBLE_DEVICES=0
# export Name="${CUDA_VISIBLE_DEVICES}-casestudy"
# echo $Name
for i in {1,2,...,10}
$Name = "重复测试/WADI学习/${i}+casestudy"
do nohup python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD \
    >> nohp-out/$Name.out 2>&1 &
    # wait $!
done
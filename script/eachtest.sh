#!/bin/zsh    
# eval "$(conda activate loranomaly  hook)"



# export Name='newencoder交环stacking和linear位置'
# export Name='ablation/no-attmamba'

#*目前最优的就是这个'attnmamba+blockize'
# export MODE='train-test'
export MODE='test'
export LOAD=None 


#*singlestep test
# export data='SMAP'
# export data='MSL'
# export data='WADI'
export data='SWaT'
#  python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD 
#* 循环使用

 
export CUDA_VISIBLE_DEVICES=1
# for data in {'SMAP','SWaT'}
for i in {1..5}
    do nohup python main.py --dataset $data \
                            --name $Name  \
                            --trainmode $MODE \
                            --loadpath $LOAD \
        >> nohp-out/$Name.out 2>&1 &
    pid=$!
    wait $pid
    # sleep 1
done

#MSL 11G,      1877
#SMAP 7G, 
#SWAT     14g, 17
#WADI 
#
#最核心的本来就是提升MSL的精度所以可以试一试？？
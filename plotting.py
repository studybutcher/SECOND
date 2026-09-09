import matplotlib.pyplot as plt
import scienceplots
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

plt.style.use(['science', 'ieee'])
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)


def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    # print('box/y',box.shape,y.shape)
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plotter(name, data_value, y_pred = None, ascore = None, labels = None,th=None,figureset=(10,2),index=True):
    """data_value: 真实的时间序列数据。
        y_pred: 模型预测的时间序列标签。
        ascore: 异常分数，表示每个时间点的异常程度。
        labels: 真实的异常标签，标识哪些时间点是异常。"""
    data_value = np.roll(data_value, 1, 0)##*这里滚动的意义是为了让每个批次来滚动，这个和这里貌似还是不同？
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    savepdf = f'plots/{name}/output.pdf'
    
    pdf = PdfPages(savepdf )
    # print('VISUAL.SHAPE',data_value.shape,y_pred.shape,labels.shape,ascore.shape)
    # np.save(f'casestudy/data/{name}_value.npy', data_value); np.save(f'casestudy/data/{name}_pred.npy', y_pred); np.save(f'casestudy/data/{name}_ascore.npy', ascore); np.save(f'casestudy/data/{name}_labels.npy', labels); np.save(f'casestudy/data/{name}_th.npy', th)
    if labels!= None:
        y = np.empty_like(labels, dtype=float) 
        y.fill(np.nan)
        indices = np.where((y_pred == 1) & (labels == 1))[0]  # 识别正确预测的索引位置
        for i in indices:#把这些位置都变成1
            y[i] = 1

    for dim in range(data_value.shape[1]):
        x_v, y_p, l, a_s = data_value[:, dim], y_pred, labels, ascore#?da大小不对应需要改嘛？(47924, 55) (47854,) (47924,) (47854,)
        
        fig, ax1= plt.subplots(1, 1,figsize=(figureset[0],figureset[1]), sharex=True)#原始就是用10，2

        # ax1.set_ylabel('Value')
        
        ax1.plot(smooth(x_v), linewidth=1, label='Data') 
        
        if index != True:#后期可以写为类然后把这些复用的都写在一起
            ax1.set_yticks([])
            ax1.set_xticks([])
        else: 
            ax1.set_title(f'Dimension = {dim}')
            ax1.legend(ncol=2, bbox_to_anchor=(0.1, 1.4))

        if labels != None:
            ax3 = ax1.twinx()
            ax3.fill_between(range(len(y)), y, color='red', alpha=0.3,label='Predicted Anomaly')  # 只在正确预测的位置填充
            if index != True:
                ax3.set_yticks([])
                ax3.set_xticks([])
            else:
                ax3.legend(ncol=2, bbox_to_anchor=(0.5, 1.4))

        pdf.savefig(fig)
        plt.close()
    print('savepdf:',savepdf)
    pdf.close()

if __name__ =='__main__':
    # name= f'casestudy/data_test/可视化_MSL'
    name= f'casestudy/data/可视化_SMAP'
    # name= f'casestudy/data/可视化_SWaT'
    ind1 = 220000
    ind2 = 250000
    #*SMAP：90000-110000 不太好用感觉？220000-250000
    #
    #*SWaT:.
    
           #*70000-100000   t1.变量在保持一致的情况起伏的情况下没有大幅度变化。并且在这个异常点单从多个维度维度上看不出任何周期上的异常。但是模型能正确识别
                            #*体现了前面的多视角拆解利用的更切片的信息获得了好的效果。，才能从中多角度来感受到直观看不到的东西。并且后续的多方面关联也有用。
                        #*t2.有时间预测性，不受时间序列的变化位置影响。能正确预测结构变化
                        # *说明在除了利用单一维度信息，在利用多个维度的相关性的基础上感知到了更多的信息。
          
         
           #*270000-None# t3在多的时间点都是正常时序，只有某一结构上产生不同变化。
           # *一定是都能体现变量的关注度不同，能分辨出来当下时刻异常是由什么变量主导的。并不是当有变化和差异时就识别异常。本质
           #*体现了什么模块吗？体现了后续多方面加权的能力由什么模块带来的效果。 在利用不同变量相关性的前提下能学习到时序状态的连续性。

    #*其他部分？
    data_value=np.load(f'{name}_value.npy')[ind1:ind2]
    y_pred=np.load(f'{name}_pred.npy')[ind1:ind2]
    ascore=np.load(f'{name}_ascore.npy')[ind1:ind2]    
    labels=np.load(f'{name}_labels.npy')[ind1:ind2]
    th=np.load(f'{name}_th.npy')
    th = np.percentile(ascore, float(th))
    # n=0
    # print(labels)
    # for q,i in enumerate(labels):
    #     if i == True:
    #         n+=1
    #         print(n,q,i)   
    # for q, i in enumerate(ascore):
    #     # print(i)
    #     if i >=th:
    #         print("fuck",q)

    # for q, i in enumerate(y_pred):
    #     if i == labels[q] and  i ==1:
    #         print("预测对了的序号",q)
        # else:
                # print("null")
  
    plotter(name, data_value, y_pred, ascore, labels,th)

import matplotlib.pyplot as plt
import scienceplots
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

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


def plotter(name, data_value, y_pred, ascore, labels,th=None):
    """data_value: 真实的时间序列数据。
        y_pred: 模型预测的时间序列标签。
        ascore: 异常分数，表示每个时间点的异常程度。
        labels: 真实的异常标签，标识哪些时间点是异常。"""
    data_value = np.roll(data_value, 1, 0)##*这里滚动的意义是为了让每个批次来滚动，这个和这里貌似还是不同？
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    # savepdf = f'plots/{name}/output.pdf'
    savepdf = f'plots/test/output.pdf'
    pdf = PdfPages(savepdf )
    # print('VISUAL.SHAPE',data_value.shape,y_pred.shape,labels.shape,ascore.shape)
    # np.save(f'casestudy/data/{name}_value.npy', data_value); np.save(f'casestudy/data/{name}_pred.npy', y_pred); np.save(f'casestudy/data/{name}_ascore.npy', ascore); np.save(f'casestudy/data/{name}_labels.npy', labels); np.save(f'casestudy/data/{name}_th.npy', th)
 
    for dim in range(data_value.shape[1]):
        x_v, y_p, l, a_s = data_value[:, dim], y_pred, labels, ascore#?大小不对应需要改嘛？(47924, 55) (47854,) (47924,) (47854,)
        
        fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(100,2), sharex=True)
        #*子图1
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        ax1.plot(smooth(x_v), linewidth=1, label='Data') 
        ax3 = ax1.twinx()
        # ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        ax3.plot(y_p, '--', color = 'blue',alpha=0.6, linewidth=1, label='Predicted')
        ax3.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.3, label='Label')#*between只能用这个
        # ax3.fill_between(np.nonzero(l)[0], 0, l[np.nonzero(l)], color='blue', alpha=0.3, label='True Anomaly')#?这个也不可以
        # anomaly_indices = np.where(l == 1)[0]
        # plt.scatter(anomaly_indices, x_v[anomaly_indices], color='red', marker='o', s=3, label='anomaly')
        
        ax1.legend(ncol=2, bbox_to_anchor=(0.1, 1.4))
        ax3.legend(ncol=2, bbox_to_anchor=(0.5, 1.4))
        
        # #*子图2
        ax2.plot(smooth(a_s), linewidth=1, color='g', label='Score')
        # ax4 = ax2.twinx()
        # #? ax4.plot(l, '--', linewidth=0.3, alpha=0.5)
        # #? ax4.fill_between(np.nonzero(l)[0], 0, l[np.nonzero(l)], color='red', alpha=0.3, label='Predicted Anomaly')
        ax2.axhline(y=th, color='red', alpha=0.3, label='Threshold')
        ax2.legend(ncol=2,bbox_to_anchor=(1, 2.6))
        # ax4.legend(ncol=2,bbox_to_anchor=(1.2, 2.6))
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        ax1.set_yticks([])
        ax3.set_yticks([])
        # ax2.set_yticks([])
        pdf.savefig(fig)
        plt.close()
    print('savepdf:',savepdf)
    pdf.close()

if __name__ =='__main__':
    # name= f'casestudy/data_test/可视化_MSL'
    # name= f'casestudy/data/可视化_SMAP'
    name= f'casestudy/data/可视化_SWaT'
    ind1 = 4000
    ind2 = 120000
    #*SWaT:270000-None.变量保持一致的情况下没有大幅度变化，但很多变量中是开始出现差异的地方，在利用不同变量相关性的前提下能学习到时序状态的连续性。
                    #*能很好的对变量加权，注意力分配的很好，才能在大段问题中完全识别。从不同变量中选出了核心问题
           #*:None-12000 另一个也是能体现加权性
           #*4000-12000这个有两个小的识别正确的点位
    #*其他部分？
    #SMAP：98000左右、220000:240000有一大段预测对了，但是不好看。需要找其他的
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

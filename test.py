
import matplotlib.pyplot as plt
import scienceplots
from scipy import stats

 #coding=utf-8
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from utils import get_data, get_data_dim, get_loader
from eval_method import bf_search
from tqdm import tqdm
import argparse
from model.Model import patchattmamba 
from model.mamba import ModelArgs

import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/Times.ttf')


plt.style.use(['science', 'ieee'])
#需要自己显示，并且之前保存的路径还是什么有问题不知道，反正不能正常调用
#并且不知道为什么只能用全称调用，这个不知道到底在调用哪里的字体格式，之前自己写的貌似都对不上#?？？还是用的自己cache的
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2
# plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.serif'] = ['Times']



def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class ExpConfig(argparse.ArgumentParser):
    # save_step=500
    val = 0.35  # the ratio of validation set
    max_train_size = None  # `None` means full train set
    train_start = 0
    max_test_size = None  # `None` means full test set
    test_start = 0
    window_length =70
    batch_size = 60
    latent_size = 256
    out_dim = 5   # the dimension of embedding
    # loadpath = '/home/yiruolei/project/anomaly-detection/SecondTask/checkpoint/0-casestudy/MSL/best.pt'
    # dataset='MSL'#:MSL,SMAP,WADI,SWaT
    # dataset='WADI'
    # dataset='SMAP'
    # dataset='SWaT'
    
    N = 256#更新步数

    def __init__(self):
        super().__init__(description="Experiment Configuration")
        # self.add_argument('--name',default= 'test',help='testname')
        # self.add_argument('--dataset',default='MSL',help='datset:MSL,SMAP,WADI,SWAT')
        # self.add_argument('--window_length',type=int,default=70)#最开始是140但是发现70更好，后面都是用的70
        # self.add_argument('--batch_size', type=int,default=60)
        # self.add_argument('--latent_size', type=int,default=256)
        # self.add_argument('--loadpath',default='None')
        self.add_argument('--loadpath',default='/home/yiruolei/project/anomaly-detection/SecondTask/checkpoint/0-casestudy/MSL/best.pt')
        self.parse_args(namespace=self)
   
class test():
     #*本身和原始代码的流程是相同的只是 这里简化了，少量数据固定模型，并且可视化数据和结构
    def __init__(self):
        self.config = ExpConfig()

    def dataload(self):
        config =self.config
        config.input_dim= get_data_dim(config.dataset)
        print('==============================================================================================')
        # print('name:',config.name)

        (train_data, _), (test_data, test_label) = \
            get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
                    test_start=config.test_start)

        n = int(test_data.shape[0] * config.val)
        test_data = test_data[:-n]
        test_label = test_label[:-n]
        
        print("test_data:", test_data.shape)
        print("test_label:", test_label.shape)
     
        return test_data,test_label 

    def detect_anomalies(self,test_data,test_label):
        # 使用Z-score方法检测异常
        # z_scores = np.abs(stats.zscore(values))
        config = self.config
        num_t = int(test_data.shape[0]/config.batch_size)
        con_t = test_data.shape[0] % config.batch_size
        w_size = config.input_dim * config.out_dim
        test_data = test_data[np.arange(config.window_length)[None, :] + np.arange(test_data.shape[0] - config.window_length)[:, None]]
        # print('zhelishisha',test_data.shape)
        test_loader = get_loader(test_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=False)
        
        argsa = ModelArgs()
        model = patchattmamba(config.input_dim, w_size, config.latent_size, config.batch_size, config.window_length, config.out_dim,args=argsa)
        model = model.cuda()
        model.load_state_dict(torch.load(config.loadpath))
        print('label',test_label.shape)#test_label: (47924,)
        test_score = model.is_anomaly(test_loader, num_t, con_t).detach().numpy()
        print('score',test_score.shape)
        _, th,y_pre= bf_search(test_score, test_label[-len(test_score):], step_num=700)

        return test_score,y_pre,th
    
    ###**和train的结构就是加了一个这个部分
def visualize_time_series_with_anomalies(data,label=0, anomalies=None,save_path=None):
    
    # plt.figure(figsize=(12, 6))
    plt.figure()##要不要先设置一个非常大的然后仔细看清楚再缩小？
    #原始时序smooth(
    plt.plot(data, label='data')
    #找出异常部分使用，而不是全部使用
    anomaly_indices = np.where(label == 1)[0]
    plt.scatter(anomaly_indices, data[anomaly_indices], color='red', marker='o', s=50, label='anomaly')
    # plt.title('')
    plt.xlabel('time')
    plt.ylabel('data')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至 {save_path}")
    plt.show()

def visualize_time_series(data, score, labels, threshold, save_path='time_series_visualization.png'):
    """
    可视化时间序列数据，标记正确和错误的识别。

    参数:
    data: numpy数组，时间序列数据
    labels: numpy数组，真实标签（0表示正常，1表示异常）
    threshold: float，判断异常的阈值
    save_path: str，保存图片的路径
    """
    # 创建预测标签
    predictions = (score > threshold).astype(int)

    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制时间序列数据
    plt.plot(data, color='blue', label='时间序列数据')
    
    # 标记阈值线
    plt.axhline(y=threshold, color='red', linestyle='--', label='阈值')

    # 标记正确识别的点
    correct = (predictions == labels)
    plt.scatter(np.where(correct & (labels == 1))[0], score[correct & (labels == 1)], 
                color='green', marker='o', s=50, label='正确识别的异常')
    plt.scatter(np.where(correct & (labels == 0))[0], score[correct & (labels == 0)], 
                color='blue', marker='o', s=50, label='正确识别的正常')

    # 标记错误识别的点
    incorrect = (predictions != labels)
    plt.scatter(np.where(incorrect & (labels == 1))[0], score[incorrect & (labels == 1)], 
                color='red', marker='x', s=50, label='未识别的异常')
    plt.scatter(np.where(incorrect & (labels == 0))[0], score[incorrect & (labels == 0)], 
                color='orange', marker='x', s=50, label='误识别的异常')

    # 设置图形标题和标签
    plt.title('时间序列异常检测可视化')
    plt.xlabel('时间')
    plt.ylabel('值')
    plt.legend()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至 {save_path}")

    # 显示图形（可选）
    # plt.show()

# 主程序
if __name__ == "__main__":
    # import time
    # print(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
    # # 生成时间序列数据
    config = ExpConfig()
    A=test()
    data, label = A.dataload()#label是false和true好像
    print(data.shape)

    
    from plotting import plotter
    name= f'开头可视化patch/{config.dataset}'
    plotter(name, data[1000:-1000,], figureset=(70,2),index= False)
    # # plotter(name,  data[-len(test_score):], y_pre, test_score, test_label[-len(test_score):],th)

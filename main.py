 #coding=utf-8
#*这个是为了超参才写的一个文件
import torch
import numpy as np
import torch.optim as optim
from utils import get_data, get_data_dim, get_loader
from eval_method import bf_search, adjust_predicts
from tqdm import tqdm
import argparse
from model.Model import patchattmamba 
from model.mamba import ModelArgs

from plotting import plotter

class ExpConfig(argparse.ArgumentParser):
    # save_step=500
    val = 0.35  # the ratio of validation set
    # max_train_size = None  # `None` means full train set
    train_start = 0#*不知道为什么要移除开头，但是可以这么设置
    max_test_size = None  # `None` means full test set
    test_start = 0

    out_dim = 5   # the dimension of embedding
    # batch_size = 60 #
    # window_length = 140
    # latent_size = 256 # the dimension of hidden layer in VAE
    N = 256#更新步数
    def __init__(self):
        super().__init__(description="Experiment Configuration")
        self.add_argument('--name',default= 'test',help='testname')
        self.add_argument('--trainmode',type=str,help='train or test')
        self.add_argument('--dataset',default='SMAP',help='datset')
        self.add_argument('--loadpath',default='None')
        self.add_argument('--window_length',type=int,default=70)#最开始是140但是发现70更好，后面都是用的70
        self.add_argument('--batch_size', type=int,default=60)#60
        self.add_argument('--latent_size', type=int,default=256)
        self.add_argument('--max_train_size', type=float,default=None)
        self.add_argument('--noise', type=float,default=0)
        self.parse_args(namespace=self)
        if self.dataset == 'SWaT':
            self.window_length = 35
        elif self.dataset == 'WADI':
            self.batch_size = 24
            # self.window_length = 140
        # elif self.dataset =='MSL':
        #     # self.window_length = 70
        #     self.latent_size=256 #?256？512貌似就不好了？
        #*下面是为了超参来设置的
        

def add_noise(data, noise_ratio):
    # 计算要替换的噪声数量
    num_samples = int(data.shape[0] * noise_ratio)
    # 随机选择要替换的索引
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    # 生成随机噪声
    # noise = np.random.normal(0, 1, data[indices].shape)
    # 生成混合噪声
    noise = np.zeros(data[indices].shape)
    for i in range(len(indices)):
        noise_type = np.random.choice(['uniform', 'spike', 'random_walk'])
        
        if noise_type == 'uniform':
            # 均匀分布噪声
            noise[i] = np.random.uniform(-2, 2, data[indices[i]].shape)
        elif noise_type == 'spike':
            # 突发性噪声
            spike_value = np.random.choice([-5, 5])
            noise[i] = np.full_like(data[indices[i]], spike_value)
        else:
            # 随机游走噪声
            noise[i] = np.cumsum(np.random.uniform(-0.5, 0.5, data[indices[i]].shape))
    # 替换数据中的部分值为噪声
    data[indices] = noise#*可以使用数组形式直接全替换。
    return data




def main():
    config = ExpConfig()
    # args = argparse.ArgumentParser()
    # args = vars(args)
    
    config.input_dim= get_data_dim(config.dataset)
    print('==============================================================================================')
    print('name:',config.name)
    print('batch:',config.batch_size)
    print('window_length',config.window_length)
    # print('hidden_size',config.hidden_size)
    print('latent_size',config.latent_size)
    print('trian or test:',config.trainmode)
    print("noise:",config.noise)

    (train_data, _), (test_data, test_label) = \
        get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
                 test_start=config.test_start)
    # 在加载数据后添加噪声

    train_data = add_noise(train_data, config.noise)

    n = int(test_data.shape[0] * config.val)
    train_data0 = train_data
    test_data = test_data[:-n]
    test_data0 = test_data
    test_label = test_label[:-n]
    val_data = test_data[-n:]
    val_data0 = val_data
    val_label = test_label[-n:]


    print("test_data:", test_data.shape)
    print("val_data:", val_data.shape)
    #*窗口化了
    train_data = train_data[np.arange(config.window_length)[None, :] + np.arange(train_data.shape[0] - config.window_length)[:, None]]#里面用来生成行标和列标。
    val_data = val_data[np.arange(config.window_length)[None, :] + np.arange(val_data.shape[0] - config.window_length)[:, None]]
    test_data = test_data[np.arange(config.window_length)[None, :] + np.arange(test_data.shape[0] - config.window_length)[:, None]]


    num_val = int(val_data.shape[0]/config.batch_size)#分几次能加载完
    con_val = val_data.shape[0] % config.batch_size #最后剩余的
    num_t = int(test_data.shape[0]/config.batch_size)
    con_t = test_data.shape[0] % config.batch_size

    w_size = config.input_dim * config.out_dim
    # print(w_size)

    train_loader = get_loader(train_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=True)
    val_loader = get_loader(val_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=True)
    test_loader = get_loader(test_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=False)

    argsa = ModelArgs()
    # print('mamba的潜在层数',argsa.d_model)
    model = patchattmamba(config.input_dim, w_size, config.latent_size, config.batch_size, config.window_length, config.out_dim,args=argsa)
    ##?因为这个导致的问题？？
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    import os
    checkpoint = os.path.join('checkpoint',f'{config.name}/{config.dataset}')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    save_path = os.path.join(checkpoint,'best.pt')
    
    if 'train' in config.trainmode:
        flag = 0
        f1 = -1
        for epoch in tqdm(range(10)):
            l = 0
            i = 0
            # for inputs in tqdm(train_loader):
            for inputs in train_loader:#*因为去了图网络跑得太快了也
                # print('input:',inputs.shape)
                inputs = inputs.cuda()
                loss = model(inputs)
                loss.backward()
                if(i% config.N == 0):#*每256步才更新一次？
                    optimizer.step()
                    optimizer.zero_grad()
                i += 1
            
            #*内存不够了，所以不用这个
            # torch.torch.save(model.state_dict(), os.path.join(checkpoint,str(epoch)+'.pt'))
            
            if(flag == 1):
                model.load_state_dict(torch.load(save_path))
            val_score = model.is_anomaly(val_loader, num_val, con_val)
            t, th,_ = bf_search(val_score, val_label[-len(val_score):], step_num=700)
            if(t[0] > f1):
                f1 = t[0]
                torch.torch.save(model.state_dict(), save_path)
                flag = 1
    
    if 'test' in config.trainmode:
        if config.loadpath == 'None':#？在参数中写入的是str结构，所以这里。
            config.loadpath = save_path
        
        # print(config.loadpath)
        model.load_state_dict(torch.load(config.loadpath))
        test_score = model.is_anomaly(test_loader, num_t, con_t)
        t, th,_ = bf_search(test_score, test_label[-len(test_score):], step_num=700)
        print('*************')
        print('dataset:', config.dataset)
        print('testname:',config.name)
        print('th:', th)
        print("TP:", t[3])
        print("FP:", t[5])
        print("FN:", t[6])
        print("precision:", t[1], "recall:", t[2], "f_score", t[0])

        return t[1], t[2], t[0]
    
    if 'visual' in config.trainmode:
        # if config.loadpath == 'None':#？在参数中写入的是str结构，所以这里。
        #     config.loadpath = save_path
        print(config.loadpath)
        model.load_state_dict(torch.load(config.loadpath))
        
        loader = test_loader
        data0= test_data0
        label = test_label
        # loader = val_loader
        # data0= val_data0
        # label = val_label
 
        score = model.is_anomaly(loader, num_t, con_t).detach().numpy()
        t, th,y_pre = bf_search(score, label[-len(score):], step_num=700)##?只用来预测最后几个的，相当于预测本身???算是自回归吗？
        # print('score',test_score.shape)#(47854,)每个都只有一个值对于这个长度的最后一个来判断？大小就是总体被判断的大小
        # print('y_pre',y_pre.shape)
        # print('newshape', test_data0[-len(test_score):].shape,test_label[-len(test_score):].shape)
        print("F1,threshold:",t[0],th)
        plotter(f'{config.name}_{config.dataset}', data0[-len(score):], y_pre, score, label[-len(score):],th)#第三个原本是loss？？这也可以？？
    
    

if __name__ == '__main__':
   main()

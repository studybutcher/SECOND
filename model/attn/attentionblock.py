import torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
# from sklearn.metrics import *
# from transformers import BertModel
# from transformers import RobertaModel
# from utils.loss import *
# from utils.utils import data2gpu, Averager, metrics, Recorder, batch_mini2sub


#*最外层的stacking框架
#*中间有很多block是加入resnet和activate func的不同形式框架
#*关于最里面的attn计算，也有不同形式的多头结构



class NormedLinear(nn.Module):#!这个不能在中间层用,太影响精度不如直接mlp
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out




class selfattentionblock(nn.Module):
    """都是对一维隐变量?[batchsize,hiddendim]"""
    def __init__(self,head:int ,indim:int,feeddim:int,layer_norm_eps=1e-5):
        super().__init__()
        self.layer1=MultiHeadedAttention(h=head,d_model=indim)#?貌似中间多加了一层全连接,,并且这个会影响数据的结构?怎么多了一层中间层?
        # self.layer1=nn.modules.activation.MultiheadAttention(num_heads=head,embed_dim=indim)
        # self.bn1 = nn.BatchNorm(indim)
        self.ln1 = nn.LayerNorm(indim,eps=layer_norm_eps)#? eps=layer_norm_eps
        self.ln2 = nn.LayerNorm(indim,eps=layer_norm_eps)#? eps=layer_norm_eps
        self.linear1 = nn.Linear(indim, feeddim)
        self.linear2 = nn.Linear(feeddim, indim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
    
     # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        return self.dropout(x)
    
    def forward(self,x):
        # print('1',x.shape)
        x = self.ln1(x+self.layer1(x,x,x)[0])#*少了一个droupout
        # print('2',x.shape)
        x = self.ln2(x+self._ff_block(x))
        # print('3',x.shape)
        return x
    


class mutualattentionblock(nn.Module):
    """都是对一维隐变量?[batchsize,hiddendim]"""
    def __init__(self,head,indim,feeddim,layer_norm_eps=1e-5):
        super().__init__()
        self.layer1=MultiHeadedAttention(h=head,d_model=indim)#?貌似中间多加了一层全连接,,并且这个会影响数据的结构?怎么多了一层中间层?
        self.layer2=MultiHeadedAttention(h=head,d_model=indim)
        self.ln1 = nn.LayerNorm(indim,eps=layer_norm_eps)#? eps=layer_norm_eps
        self.ln2 = nn.LayerNorm(indim,eps=layer_norm_eps)
        self.ln3 = nn.LayerNorm(indim,eps=layer_norm_eps)
        self.linear1 = nn.Linear(indim, feeddim)
        self.linear2 = nn.Linear(feeddim, indim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
    
     # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        return self.dropout(x)
    
    def forward(self,x,mutualinfo):
        # print('1',x.shape)
        x = self.ln1(x+self.layer1(x,x,x)[0])

        x = self.ln2(x+self.layer2(mutualinfo,mutualinfo,x)[0])#*少了一个droupout
        # print('2',x.shape)
        x = self.ln3(x+self._ff_block(x))
        # print('3',x.shape)
        return x
    

class selfattentionblock1d(nn.Module):
    """都是对一维隐变量?[batchsize,hiddendim]"""
    def __init__(self,head,indim,feeddim,layer_norm_eps=1e-5):
        super().__init__()
        # self.layer1=MultiHeadedAttention(h=head,d_model=indim)#?貌似中间多加了一层全连接,,并且这个会影响数据的结构?怎么多了一层中间层?
        self.layer1=nn.modules.activation.MultiheadAttention(num_heads=head,embed_dim=indim)
        # self.bn1 = nn.BatchNorm(indim)
        self.ln1 = nn.LayerNorm(indim,eps=layer_norm_eps)#? eps=layer_norm_eps
        self.ln2 = nn.LayerNorm(indim,eps=layer_norm_eps)#? eps=layer_norm_eps
        self.linear1 = nn.Linear(indim, feeddim)
        self.linear2 = nn.Linear(feeddim, indim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
    
     # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        return self.dropout(x)
    
    def forward(self,x):
        # print('1',x.shape)#1 torch.Size([120, 70, 25])#?为什么是这个大小，太奇怪了
        x = self.ln1(x+self.layer1(x,x,x)[0])#*少了一个droupout
        # print('2',x.shape)
        x = self.ln2(x+self._ff_block(x))
        # print('3',x.shape)
        return x
    


class attentionstacking(nn.Module):
    """一个堆叠类,只要block输入输出相同就可以用这个来堆叠"""
    def __init__(self,block:nn.Module,head,indim,feeddim,layer_norm_eps=1e-5):
        super().__init__()
        self.selfblock1= block(head=head,indim=indim,feeddim=feeddim)
        self.selfblock2= block(head=head,indim=indim,feeddim=feeddim)
        self.selfblock3= block(head=head,indim=indim,feeddim=feeddim)
        self.ln = nn.LayerNorm(indim,eps=layer_norm_eps)#? eps=layer_norm_eps
    
    def forward(self,x):
        """san步一残差"""
        # print(x)
        # x= self.selfblock(x)
        x = self.ln(x+self.selfblock1(x))
        x= self.selfblock2(x)
        # x = self.ln(x+self.selfblock(x))
        x= self.selfblock3(x)
   
        return x
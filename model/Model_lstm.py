import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import softplus
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from model.embed import *
# from loss import CenterLoss
from .attn.attentionblock import *
from .mamba import ModelArgs
from .backbone import LSTMMixerBlock
import math
import time


from tqdm import tqdm
from torch.autograd import Variable
from einops import rearrange,repeat,reduce


def tabular_encoder(input_size: int, latent_size: int):
    """
    Simple encoder for tabular data.
    If you want to feed image to a VAE make another encoder function with Conv2d instead of Linear layers.
    :param input_size: number of input variables
    :param latent_size: number of output variables i.e. the size of the latent space since it's the encoder of a VAE
    :return: The untrained encoder model
    """
    return nn.Sequential(#*因为进行了插值所以有变化
        #input_size =w_size,不是window
        nn.Linear(input_size, latent_size * 2),
        attentionstacking(selfattentionblock1d,head=8,indim=latent_size * 2,feeddim=1024),
        attentionstacking(selfattentionblock1d,head=8,indim=latent_size * 2,feeddim=1024),#调参增加层数
        attentionstacking(selfattentionblock1d,head=8,indim=latent_size * 2,feeddim=1024),
        
        # times 2 because this is the concatenated vector of latent mean and variance
    )


def tabular_decoder(latent_size: int, output_size: int):
    """
    Simple decoder for tabular data.
    :param latent_size: size of input latent space
    :param output_size: number of output parameters. Must have the same value of input_size
    :return: the untrained decoder
    """
    return nn.Sequential(
        attentionstacking(selfattentionblock1d,head=8,indim=latent_size,feeddim=1024),
        attentionstacking(selfattentionblock1d,head=8,indim=latent_size,feeddim=1024),
        nn.Linear(latent_size, output_size * 2),

        # times 2 because this is the concatenated vector of reconstructed mean and variance
    )


class patchattmamba(nn.Module):

    def __init__(self, input_size: int, w_size, latent_size, batch_size, window_length, out_dim,dropout=0.0, args=None):
        """
        :param input_size: Number of input features,维度
        :w_size 是啥玩意？？#?这个不是哪个大小
        :param window_length:这个才是输入窗口大小
        :param latent_size: Size of the latent space
        :param L: Number of samples in the latent space (See paper for more details)
        """
        super().__init__()
        # self.patch_size = patch_size
        self.patch_size = [5,7]
        self.input_size = input_size
        self.w_size = w_size#channel*设置的输出维度
        self.latent_size = latent_size
        self.batch_size = batch_size
        # self.hidden_size = hidden_size

       
        self.embedding_inpatch = nn.ModuleList()
        self.embedding_patchwise = nn.ModuleList()
        self.attentiona = nn.ModuleList()
        self.attentionb = nn.ModuleList()
        self.projection1 = nn.ModuleList()
        self.projection2 = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            patchnum=window_length //patchsize
            self.embedding_inpatch.append(DataEmbedding(patchsize, self.latent_size, dropout))#?没设置dropout
            self.embedding_patchwise.append(DataEmbedding(patchnum, self.latent_size, dropout))
            # if self.input_size == 25:
            #     head = 5
            # elif self.input_size == 51: 
            #     head = 3
            # elif self.input_size == 127: 
            #     head = 1
            # elif self.input_size == 55: 
            #     head = 5
            # self.attentiona.append(attentionstacking(selfattentionblock,head=4,indim=self.latent_size,feeddim=512))
            # self.attentionb.append(attentionstacking(selfattentionblock,head=4,indim=self.latent_size,feeddim=512))
            self.projection1.append(nn.Linear(patchnum*self.latent_size,self.latent_size))#?这个可能每个实验都要改？怎么用其他变量转化
            self.projection2.append(nn.Linear(patchsize*self.latent_size,self.latent_size))

        
        # self.attention_C = attentionstacking(selfattentionblock,head=4,indim=512,feeddim=256)
        #todo 把内外层的多少结构反过来用试一试？
        # self.attention_C = attentionstacking(selfattentionblock1d,head=4,indim=512,feeddim=1024)
        #*simple输入
        feeddim=1024
        indim = self.latent_size*2
        self.simpembdding = DataEmbedding(window_length, indim, dropout)
        self.project4 = nn.Linear(input_size*indim,indim)
        self.attention_sim = attentionstacking(selfattentionblock1d,head=4,indim=indim,feeddim=feeddim)
        

        self.attention_ALL= attentionstacking(selfattentionblock1d,head=8,indim=indim,feeddim=feeddim)
        self.projection3 = nn.Linear(indim,w_size)
        self.encoder = tabular_encoder(w_size, latent_size)
        self.decoder = tabular_decoder(latent_size, w_size)

        
        # args.d_model= 1024#这个来不及
        args.d_model = indim#用来拼接
        self.attmamba = nn.ModuleList([LSTMMixerBlock(args) for _ in range(args.n_layer)])

        self.criterion=nn.CosineSimilarity(dim=-1)
        self.prior = Normal(0, 1)

        ##*test
        
        

        # self.attention_test = attentionstacking(selfattentionblock1d,head=5,indim=125,feeddim=1024)

    def forward(self, x):
        x_a,pred_result,loss_c = self.pipline(x)
        # x_a,pred_result,loss_c= self.simpip(x)
        mu = pred_result['recon_mu']
        # print(mu.shape)#[batch 125]
        # Xt = Xt.view(-1, Xt.shape[1] * Xt.shape[2])
        loss_function = torch.nn.MSELoss()
        l1 = loss_function(mu, x_a)#重构损失？直接用这个基本的就行？
        l2 = torch.mean(-0.5 * torch.sum(1 + pred_result['latent_sigma'] - pred_result['latent_mu'].pow(2) - pred_result['latent_sigma'].exp(),
                                          dim=1),
                         dim=0)#kl散度
        loss = l1 + l2 +loss_c
        # timefinal=time.time()

        return loss

    def is_anomaly(self, x, num_t, con_t):
        """

        :param x:
        :param alpha: Anomaly threshold (see paper for more details)
        :return: Return a vector of boolean with shape [x.shape[0]]
                 which is true when an element is considered an anomaly
        """
        score = None##
        for i, inputs in enumerate(x, 0):#太快了取消tqdm？
             #input是一个batch数据
            # print('input',inputs.shape)#[batch win channel]
            inputs = inputs.cuda()
            p = self.reconstructed_probability(inputs)
            # print('p',p.shape) #[batchsize]
            if i==num_t:#加载到最后一个batch
                p = p[:con_t]
            if score is None:
                score = p
            else:
                score = torch.cat((score, p), dim=0)
        # score = torch.tensor(score)
        return score

    def reconstructed_probability(self, x):
        with torch.no_grad():
            x_a,pred_result,_ = self.pipline(x,'test')
            # x_a,pred_result,_ = self.simpip(x,'test')
            mu = pred_result['recon_mu']
            p = []
            for i in range(x.shape[0]):
                t = abs(torch.sum((x_a[i] - mu[i])))#*重建的和原本的差值就是具体分数。本质就是对所有数据进行排序或者二聚类。然后用动态选择阈值来把所有数据分开。
                # print(t)#0维数据？
                p.append(t)
        # print('1',p.shape)
        p = torch.tensor(p)###*这个貌似没有？
        return p
########################

    def data_patch(self,x,patchsize,):
        x_inpatch, x_patchwise = x, x#x [batch win_size channel]
        x_inpatch = rearrange(x_inpatch, 'b l m -> b m l') #Batch channel win_size
        x_patchwise = rearrange(x_patchwise, 'b l m -> b m l') #Batch channel win_size
        
        x_inpatch = rearrange(x_inpatch, 'b m (n p) -> (b m) n p', p = patchsize) ##对patch内数据边话
        x_patchwise = rearrange(x_patchwise, 'b m (p n) -> (b m) p n', p = patchsize) ##？？这算是对batch整体吗？
        return x_inpatch,x_patchwise
    
    def pipline(self,x,mode=None):
        # B,W,D=x.shape#torch.Size([120,win , 25])
        #*win内时间不同尺度上的embeding，把batch和channel放在一起是不直接使用的，后续也好分离出来
        pre_inpatch_mean=[]
        pre_patchwise_mean=[]
        # time0=time.time()
        for patch_index, patchsize in enumerate(self.patch_size):
            # patchnum = W//patchsize
            x_inpatch,x_patchwise= self.data_patch(x,patchsize)#patch_num*patch_size=winnumber
            # print(x_inpatch.shape,x_patchwise.shape)
            #下面这两个也是只改变最后一个
            x_inpatch = self.embedding_inpatch[patch_index](x_inpatch)#[batchsize*channel patchnum dim]
            x_patchwise= self.embedding_patchwise[patch_index](x_patchwise)#[batchsize*channel patch_size dim]
            
            # x_inpatch = self.attentiona[patch_index](x_inpatch)
            # x_patchwise = self.attentionb[patch_index](x_patchwise)
           
            x_inpatch = rearrange(x_inpatch,'b p q  -> b  (p q)')#[batchsize*channel patchnum*dim]
            x_patchwise = rearrange(x_patchwise,'b p q -> b (p q)')#[batchsize*channel patch_size*dim]

            x_inpatch = self.projection1[patch_index](x_inpatch)#[batchsize*channel dim]
            x_patchwise = self.projection2[patch_index](x_patchwise)#[batchsize*channel dim]
            #*这个loss先不改把就这样，毕竟精度也还可以
            loss_c=(self.criterion(x_inpatch, x_patchwise ).mean() + self.criterion(x_patchwise , x_inpatch).mean()) * 0.5#因为目标是特征差异化，所以相似度变小？不行了就换方向试一试
            #这里要求均值
            pre_inpatch_mean.append(x_inpatch), pre_patchwise_mean.append(x_patchwise)

        pre_inpatch_mean = torch.stack(pre_inpatch_mean,dim=-1).mean(dim=-1)
        pre_patchwise_mean = torch.stack(pre_patchwise_mean,dim=-1).mean(dim=-1)
        # print(pre_inpatch_mean.shape,pre_patchwise_mean.shape)
        #*拼接
        x_combined = torch.cat((x_inpatch, x_patchwise), dim=-1)  # 拼接两个变量，[batchsize*channel dim*2]#?先拼还是先变化？之后试一试
        x_combined = rearrange(x_combined,'(a b) c -> a b c',b = self.input_size)#*补充了之前没有channel用来学习的部分
        # print('2',x_combined.shape)#[batchsize channel dim*2]
        # time1=time.time()
        #*attmamba#从这里开始一样，前面得优化一下得到最好效果
        for layer in self.attmamba:
            x_combined = layer(x_combined)
        # time2=time.time()
        # print('1',x_combined.shape)
        #!最后一次操作，之后不直接平均，用投影的方式再一次学习试一试这个？
        x = rearrange(x_combined,'b p q -> b (p q)')# [120, channel*dim*2]
        # print(x.shape)
        x = self.project4(x)#[batch 2dim]
        # x_combined  = reduce(x_combined, 'b reduce_b m -> b m', 'mean', reduce_b=self.input_size)#[batchsize, 2dim]
        ##*后面是patch后相同的操作
        x = self.attention_sim(x)#[batch 2dim]
        x_all = self.attention_ALL(x)##[batch 2dim]
        x_a = self.projection3(x_all)#[batchsize w_dim]#?可以后面改
        # print(x_a.shape)
        # time3=time.time()
        pred_result = self.predict(x_a)
            
        # print("part1:{},part2:{},all:{}".format(time1-time0,time2-time1,time3-time0))
        return x_a,pred_result,loss_c
    
    def simpip(self,x,mode=None):
        # B,W,D=x.shape#torch.Size([120,win , channel])
        # print(x.shape)
        x = rearrange(x,'b p q -> b q p')# [120, channel win]
        x= self.simpembdding(x) #[120,channel，dim]
        # print(x.shape)
        for layer in self.attmamba:#?貌似没并行？为什么这么慢
            x = layer(x)
        # print(x.shape)
        x = rearrange(x,'b p q -> b (p q)')# [120, channel*dim]
        x = self.project4(x)
        x= self.attention_sim(x)#[batchsize 512]
        x_all = self.attention_ALL(x)#[b 256]
        x_a = self.projection3(x_all)#[batchsize w_dim]#?可以后面改
        pred_result = self.predict(x_a)
            
        return x_a,pred_result,0

    def predict(self, x) -> dict:
        """
        :param x: tensor of shape [batch_size, num_features]
        :return: A dictionary containing prediction i.e.
        - latent_dist = torch.distributions.Normal instance of latent space
        - latent_mu = torch.Tensor mu (mean) parameter of latent Normal distribution
        - latent_sigma = torch.Tensor sigma (std) parameter of latent Normal distribution
        - recon_mu = torch.Tensor mu (mean) parameter of reconstructed Normal distribution
        - recon_sigma = torch.Tensor sigma (std) parameter of reconstructed Normal distribution
        - z = torch.Tensor sampled latent space from latent distribution
        """
        batch_size,D=x.shape
        ## 设置mask，mask一部分的数据，然后通过几层后恢复结构
        #*这里没有设置mae的形式，基本的单纯mask
        # mask = torch.ones_like(x)
        # mask[:, :W//2, :] = 0  # mask掉前一半的数据
        # x_masked = x * mask   

        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1)
        latent_sigma = softplus(latent_sigma) + 1e-4
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample()
        z = z.view(batch_size, self.latent_size)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        # # 恢复结构
        # x_restored = x_masked + (x * (1 - mask))
        recon_sigma = softplus(recon_sigma) + 1e-4
        recon_mu = recon_mu.view(-1, x.shape[1])
        recon_sigma = recon_sigma.view(-1, x.shape[1])

        return dict(latent_dist=dist, latent_mu=latent_mu,
                latent_sigma=latent_sigma, recon_mu=recon_mu,
                recon_sigma=recon_sigma, z=z)
     
    def predict_masked(self, x) :
    #这个懒得看了，会自动生成-> (dict, int)
        """
        :param x: tensor of shape [batch_size, num_features]
        :return: A dictionary containing prediction i.e.
        - latent_dist = torch.distributions.Normal instance of latent space
        - latent_mu = torch.Tensor mu (mean) parameter of latent Normal distribution
        - latent_sigma = torch.Tensor sigma (std) parameter of latent Normal distribution
        - recon_mu = torch.Tensor mu (mean) parameter of reconstructed Normal distribution
        - recon_sigma = torch.Tensor sigma (std) parameter of reconstructed Normal distribution
        - z = torch.Tensor sampled latent space from latent distribution
        """
        batch_size,D=x.shape
        ## 设置mask，mask一部分的数据，然后通过几层后恢复结构
        #*这里都是按照只有一层dim来看的，多维的其他位置要小改一下
        mask_rat=D//9
       
        mask = torch.ones_like(x)
        mask_indices = torch.randperm(D)[:mask_rat]  # 随机选择一半的数据进行mask
        mask[:, mask_indices] = 0  # mask掉随机选择的部分数据
        #?使用inf还是会导致梯度和tensor出问题
        # x = x.masked_fill(mask == 0, float("-inf"))
        # delate_data = x.masked_fill(mask == 1, float("-inf"))
        # print(x,delate_data)
        # ?改成scores试一试？
        delate_data = x *(1-mask) 
        x = x * mask
        # print(delate_data,x)
        # print(delate_data.shape,x.shape)
        # x = x[mask.bool()].view(batch_size, -1)  # 仅保留 mask 为 1 的部分
        #*mask.bool()这个也非常好用
        
              
        #*unmask支路
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1)
        latent_sigma = softplus(latent_sigma) + 1e-4
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample()
        z = z.view(batch_size, self.latent_size)
        #*mask支路
        mask_mu, mask_sigma = self.encoder(delate_data).chunk(2, dim=1)
        mask_sigma = softplus(mask_sigma) + 1e-4
        # maskdist = Normal(mask_mu, mask_sigma)
        # maskz = maskdist.rsample()
        # maskz = maskz.view(batch_size, self.latent_size)

        loss_masker = torch.nn.MSELoss()
        loss_mask1 = loss_masker(latent_mu, mask_mu)
        # loss_mask2=0
        loss_mask2 = loss_masker(latent_sigma, mask_sigma)+loss_mask1

        # unmask = torch.where(mask.bool(), z, delate_data)  # 按照mask=1填入z，mask=0填入原始delate_data
        # unmask[mask.bool()]=z
        # z_re = z + (x * (1 - mask))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        # # 恢复结构
        
        recon_sigma = softplus(recon_sigma) + 1e-4
        recon_mu = recon_mu.view(-1, x.shape[1])
        recon_sigma = recon_sigma.view(-1, x.shape[1])

        return dict(latent_dist=dist, latent_mu=latent_mu,
                latent_sigma=latent_sigma, recon_mu=recon_mu,
                recon_sigma=recon_sigma, z=z), loss_mask2



    
    

    def generate(self, batch_size: int=1) -> torch.Tensor:
        """
        Sample from prior distribution, feed into decoder and get in output recostructed samples
        :param batch_size:
        :return: Generated samples
        """
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)



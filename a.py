# # encoding: utf-8
# import torch
import numpy as np
# import pandas as pd
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print('使用的设备是:', device)
# # np.set_printoptions(threshold=np.inf)#可以让接下来所有的np输出不省略
# #重新省略就设置threshold为多少个现实值
# path='/home/yiruolei/ALLDATASET/anomaly/WADI/WADI_test_label.npy'
# # path='/home/yiruolei/ALLDATASET/anomaly/SWaT/SWaT_train.npy'
# a = np.load(path)
# a = a[:,0]
# with open('/home/yiruolei/ALLDATASET/anomaly//WADI_test_label2.npy', "wb") as file:
#             np.save(file,a)
# # Df = pd.DataFrame(a[0])
# # file_csv = Df.to_csv('./processed/SWat_label.csv') 
# print(a.shape)

# import matplotlib as plt
# import matplotlib.font_manager as fm
# # fm.fontManager.addfont('/home/yiruolei/.config/matplotlib/Times New Roman.ttf')

# font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
# for font_path in font_paths:
#   print(font_path)
# print(plt.get_cachedir())
# #    print(matplotlib.get_configdir())

a = np.load('a.npy')
b = np.load('b.npy')
# if a ==b:
#   print('yes')
# else:
#   print('fuck')
print(a.shape,b.shape)
print(a,b)

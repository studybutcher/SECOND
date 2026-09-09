import ast
import csv
import os
import sys
from pickle import dump
import pandas as pd

import numpy as np
# from tfsnippet.utils import makedirs

# output_folder = 'processed'
output_folder = '/home/yiruolei/ALLDATASET/anomaly'
# makedirs(output_folder, exist_ok=True)


def load_and_save(category, filename, dataset, dataset_folder):
    #读取本地文件
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')  #由逗号分离
    print(dataset, category, filename, temp.shape)
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset,root=None):
    if dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = root+'SMAP_MSL'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]    #一行行的列表
        res = sorted(res, key=lambda k: k[0]) #按列表中第一个元素排序
        label_folder = os.path.join(dataset_folder, 'test_label')
        makedirs(label_folder, exist_ok=True)  #递归创建目录
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])  #对字符串进行类型转换
            length = int(row[-1])   #-1表示最后一行的数据
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels) #转换成数组
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ['train', 'test']:
            concatenate_and_save(c)

    elif dataset == 'SWaT':
        dataset_folder = os.path.join(root,'SWAT')
        # normal = pd.read_csv(os.path.join(dataset_folder,"SWaT/SWaT_Dataset_Normal_v1.csv")) #, nrows=1000)
        # normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
        train_data = pd.read_csv(os.path.join(dataset_folder, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(dataset_folder, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]
        
        # train_data = np.asarray(train_data)
        print(dataset, 'train', train_data.shape)
        # test_data = np.asarray(test_data)
        print(dataset, 'test', test_data.shape)
        # labels = np.asarray(labels)
        print(dataset, 'labels', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'train' + ".npy"), "wb") as file:
            np.save(file,train_data)
        # np.save(train_data, os.path.join(output_folder, dataset + "_" + 'train' + ".npy"))
        with open(os.path.join(output_folder, dataset + "_" + 'test' + ".npy"), "wb") as file:
            np.save(file,test_data )
        # np.save(test_data, os.path.join(output_folder, dataset + "_" + 'test' + ".npy"))
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".npy"), "wb") as file:
            np.save(file,labels)
        # np.save(labels, os.path.join(output_folder, dataset + "_" + 'test_label' + ".npy"))

    elif dataset == 'WADI':
        dataset_folder = os.path.join(root,'data')
        normal = pd.read_csv(os.path.join(dataset_folder,"WADI/WADI_14days.csv"),sep=',',error_bad_lines=True,)
        # normal = pd.read_csv(os.path.join(root,"WADI/WADI_14days_new.csv"))#*没用上new怎么办？
        normal = normal.drop(['Row', 'Date', 'Time'], axis=1)
        normal = np.asarray(normal)
        print(dataset, 'train', normal.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'train' + ".npy"), "wb") as file:
            np.save(normal, file)

        attack = pd.read_csv('WADI/WADI_attackdataLABLE.csv')
        labels = [float(label != 1) for label in attack['Attack LABLE (1:No Attack, -1:Attack)'].values]
        attack = attack.drop(['Row', 'Date', 'Time', 'Attack LABLE (1:No Attack, -1:Attack)'], axis=1)
        labels = np.asarray(labels)
        print(dataset, 'labels', labels.shape)
        attack = np.asarray(attack)
        print(dataset, 'test', attack.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test' + ".npy"), "wb") as file:
            np.save(attack, file)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".npy"), "wb") as file:
            np.save(labels, file)


if __name__ == '__main__':
    datasets = ['SMAP', 'MSL', 'SWaT', 'WADI']
    root = '/home/yiruolei/ALLDATASET/anomaly/'
    # datasets = ['WADI']
    commands = sys.argv[1:]#?还不会用
    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d,root )
    else:
        print("""
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMAP', 'MSL', 'SWaT', 'WADI']
        """)

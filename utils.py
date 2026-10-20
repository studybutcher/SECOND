import os
import json
import pickle
import torch
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

prefix = "/data/sdc/yrlbkp/anomaly"

# PRD7: HAI 22.04 / 23.05。npy 存清洗后的原始值，归一化在加载时用 scale.json。
HAI_DATASETS = {'HAI22', 'HAI23'}
HAI_DIM = 78

def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif dataset == 'SWaT':
        return 51
    elif dataset == 'WADI':
        return 127
    elif dataset in HAI_DATASETS:
        return HAI_DIM
    else:
        raise ValueError('unknown dataset '+str(dataset))


def load_hai_scale(dataset):
    """读 scale.json，返回与 78 列同序的 (min, max, degenerate) 数组。"""
    path = os.path.join(prefix, dataset, 'scale.json')
    with open(path) as f:
        meta = json.load(f)
    stats = meta['stats']
    mn = np.array([s['min'] for s in stats], dtype=np.float32)
    mx = np.array([s['max'] for s in stats], dtype=np.float32)
    dg = np.array([bool(s['degenerate']) for s in stats])
    return mn, mx, dg


def apply_minmax(X, mn, mx, dg):
    """源域 MinMax + 零范围保护（PRD7 §2.6）。

    源域 train 上 max==min 的列，在所有 split、所有域里一律输出 0。
    """
    if X is None:
        return None
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] != len(mn):
        raise ValueError(f'列数 {X.shape[1]} != scale.json 的 {len(mn)}')
    den = np.where(dg, 1.0, mx - mn).astype(np.float32)
    Y = (X - mn) / den
    if dg.any():
        Y[:, dg] = 0.0
    return Y


def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0, minmax_from=None, load_train=True):
    """
    get data from pkl files

    minmax_from : PRD7 交叉评测用。HAI 数据集加载时用 `minmax_from or dataset`
                  的 scale.json 做源域 MinMax；目标域 test 必须传源域名。
    load_train  : False 时跳过 train npy 的读取，返回 (None, None)。
                  test-only 交叉评测不需要源域 train（HAI22 有 100 万行，
                  窗口化后会占 20+ GB）。

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    elif dataset in HAI_DATASETS:
        # PRD7: HAI 的 --max_train_size 解释为**绝对行数**（便于子集冒烟）。
        # 旧的 ×58317 是 MSL 专用缩放，对 HAI 无意义。
        train_end = train_start + int(max_train_size)
    else:
        max_train_size = int(max_train_size*58317)
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    if load_train:
        f = open(os.path.join(prefix,str(dataset)+'/'+str(dataset)+'_train.npy'), "rb")
        train_data = np.load(f).reshape((-1, x_dim))[train_start:train_end, :]
        f.close()
    else:
        train_data = None
    try:
        f = open(os.path.join(prefix, f'{dataset}/{dataset}_test.npy'), "rb")
        test_data = np.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, f'{dataset}/{dataset}_test_label.npy'), "rb")
        test_label = np.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if dataset in HAI_DATASETS:
        # PRD7：npy 是原始值，用**源域** train 的 scale.json 归一化。
        # 交叉评测（--test-dataset）时 minmax_from 指向源域，
        # 目标域 test 因此与源域 train/test 共用同一套 min/max。
        src = minmax_from or dataset
        mn, mx, dg = load_hai_scale(src)
        train_data = apply_minmax(train_data, mn, mx, dg)
        test_data = apply_minmax(test_data, mn, mx, dg)
        print(f'Hai MinMax from: {src}/scale.json  degenerate={int(dg.sum())}')
    elif do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

    print("train set shape: ", None if train_data is None else train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)


    return (train_data, None), (test_data, test_label)


def _clean(df):
    """NaN→0、inf→100。与 preprocess 改动前的逐字等价。"""
    df = np.asarray(df, dtype=np.float32)
    # np.set_printoptions(threshold=99999999999999999)
    # print(df)
    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        inx = np.isnan(df)
        df[inx] = 0
    if np.isinf(df).any():
        inx = np.isinf(df)
        print('Data contains inf values. Will be replaced with 100')
        df[inx] = 100
    return df


def preprocess(df, fit_data=None):
    """returns normalized and standardized data.

    fit_data 非 None 时：用 fit_data 拟合 MinMaxScaler，再 transform df
    （两者须同维度）。供异常污染实验使用——异常在**原始物理空间**注入，
    而 scaler 只能见过 clean 数据（见 docs/PRD4text_定量分析_r2.md §6 方案 B）。

    fit_data=None 时行为与改动前完全一致（scaler 在 df 上拟合）。
    """
    df = _clean(df)
    # normalize data
    scaler = MinMaxScaler().fit(df if fit_data is None else _clean(fit_data))
    df = scaler.transform(df)
    print('Data normalized')

    return df

def BatchSlidingWindow(values, window_length):
    # data = []
    # for i in range(len(values) - window_length + 1):
    #     a = []
    #     for j in range(window_length):
    #         a = np.concatenate((a, values[i+j]), axis=0)
    #     # a = np.asarray(a)
    #     data.append(a)
    # data = np.array(data)
    # # assert len(data.shape) == 3
    data = []
    for i in range(len(values) - window_length):
        data.append(values[i:i + window_length])
    data = np.array(data)
    # assert len(data.shape) == 3
    return data

def joint(values):
    data = []
    for i in range(values.shape[0]):
        a = []
        for j in range(values.shape[1]):
            a = np.concatenate((a, values[i][j]), axis=0)
        data = np.concatenate((data, a), axis=0)
    return data

def get_loader(values, batch_size, window_length, input_size, shuffle=False):
    if values.shape[0] % batch_size !=0:
        for i in range(batch_size-values.shape[0] % batch_size):
            a = torch.tensor(np.zeros((1, window_length, input_size), dtype='float32'))
            values = np.concatenate((values, a), axis=0)
    values = torch.tensor(values)
    return DataLoader(dataset=values, batch_size=batch_size, shuffle=shuffle)
    # return DataLoader(dataset=values.view(([values.shape[0], values.shape[1] * values.shape[2]])), batch_size=batch_size, shuffle=shuffle)

def load_data(f_name, f_name2):
    # print('We are loading testing data from:', f_name)
    true_edge = []
    false_edge = []
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split()
            x, y = words[0], words[1]
            true_edge.append((x, y))
    with open(f_name2, 'r') as f:
        for line in f:
            words = line[:-1].split()
            x, y = words[0], words[1]
            false_edge.append((x, y))

    return true_edge, false_edge

def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        # return np.dot(vector1, vector2)
        return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass

def GCN_Loss(emb):

    emb_true_first = []
    emb_true_second = []
    emb_false_first = []
    emb_false_second = []

    emb = emb.permute(1, 0)

    true_edges, false_edges = load_data('tmp.txt', 'tmp2.txt')
    for edge in true_edges:
        emb_true_first.append(emb[int(edge[0])].detach().numpy())
        emb_true_second.append(emb[int(edge[1])].detach().numpy())

    for edge in false_edges:
        emb_false_first.append(emb[int(edge[0])].detach().numpy())
        emb_false_second.append(emb[int(edge[1])].detach().numpy())

    T1 = np.dot(np.array(emb_true_first), np.array(emb_true_second).T)
    T2 = np.dot(np.array(emb_true_first), np.array(emb_true_second).T)

    pos_out = torch.tensor(np.diagonal(T1))
    neg_out = torch.tensor(np.diagonal(T2))

    loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

    return loss
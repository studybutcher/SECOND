 #coding=utf-8
#*这个是为了超参才写的一个文件
import json
import os
import random
import time

import torch
import numpy as np
import torch.optim as optim
from utils import get_data, get_data_dim, get_loader
from eval_method import bf_search, adjust_predicts, pot_eval, DTAAD_LM, DTAAD_Q
from tqdm import tqdm
import argparse
from model.mamba import ModelArgs

from plotting import plotter

# 消融变体：所有变体的模型类同名，仅 import 来源不同。
# 前三个是**替换型**（PRD1：测 SSM mixer 是否不可替代）；
# 后三个是**删除型**（测模块本身是否需要存在）。
# 全部由 script/addition/make_ablation_models.py 生成，禁止手工编辑。
BACKBONE_REGISTRY = {
    'mamba':       'model.Model',            # 基线（attmambaBlock）
    'transformer': 'model.Model_trans',      # 替换：中间块 mixer → Transformer
    'lstm':        'model.Model_lstm',       # 替换：中间块 mixer → 双向 LSTM
    'nomssa':      'model.Model_nomssa',     # 删除：整块拿掉 8 层中间块
    'nompa':       'model.Model_nompa',      # 删除：MPA 多视图分块 + loss_c
    'nompamssa':   'model.Model_nompamssa',  # 删除：上面两个都删
    # PRD5.2：传统 LSTM-AE，只为在 script/robust 那套加噪协议下换模型（不涉及 MPA/MSSA/CRR）
    'lstmae':      'model.Model_lstmae',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        # default=None：各数据集的默认值在下面按 dataset 补，**显式传入的命令行值优先**。
        # 旧写法是在 parse_args 之后无条件覆盖（WADI 恒为 24、SWaT 恒为 35），
        # 导致 `--batch_size 32` 被静默吃掉——历史最优的 WADI 用的是 bs=32，
        # 在那个写法下根本无法复现。见 docs/定量分析_WADI基线诊断.md
        self.add_argument('--window_length',type=int,default=None)#最开始是140但是发现70更好，后面都是用的70
        self.add_argument('--batch_size', type=int,default=None)#60
        self.add_argument('--latent_size', type=int,default=256)
        self.add_argument('--max_train_size', type=float,default=None)
        self.add_argument('--noise', type=float,default=0)
        self.add_argument('--noise-kind', type=str, default='mixed',
                          choices=['mixed', 'gauss'],
                          help='mixed=现有混合噪声（默认，保持旧结果可复现）；'
                               'gauss=基础版整行替换为 N(0,1)')
        self.add_argument('--backbone', type=str, default='mamba',
                          choices=list(BACKBONE_REGISTRY), help='中间块消融变体')
        self.add_argument('--seed', type=int, default=42, help='随机种子')
        self.add_argument('--result-dir', type=str, default=os.path.join('results', 'ablation'),
                          help='结果 JSON 落盘目录（污染实验传 results/contam，'
                               '避免覆盖 PRD1 中间块消融的结果）')
        self.add_argument('--result-name', type=str, default=None,
                          help='结果 JSON 文件名（不含 .json）；'
                               '默认 {backbone}_{dataset}_seed{seed}')
        self.add_argument('--contam-file', type=str, default=None,
                          help='污染训练集 .npy（已 preprocess，直接替换 train_data，'
                               '不再二次归一化）；见 bench/gen_contam.py')
        self.add_argument('--contam-manifest', type=str, default=None,
                          help='污染 manifest .json；默认与 --contam-file 同名')
        self.add_argument('--contam-test-file', type=str, default=None,
                          help='【路 B】测试侧污染 .npy（已 preprocess，直接替换 test_data；'
                               '训练侧保持干净）。生成：bench/gen_contam.py --split test')
        self.add_argument('--score-dump', type=str, default=None,
                          help='把 test/val 的分数与标签落盘成 .npz（PRD5.2）。'
                               '评测协议本身不变，只是为了事后能算**未调整 AUPRC**——'
                               'point-adjust F1 已实测无分辨力')
        self.add_argument('--ckpt-dir', type=str, default=None,
                          help='checkpoint 目录；默认 checkpoint/<name>/<dataset>。'
                               '⚠️ 并行的同变体 run（如 mamba 的多个 ρ/seed）必须各给一个，'
                               '否则会并发写同一个 best.pt → 读到半截文件 '
                               '（PytorchStreamReader ... central directory），且互相覆盖')
        #* ---- 阈值选择策略（POT），默认 'bf' 时行为与改动前完全一致 ----
        self.add_argument('--threshold-mode', type=str, default='bf',
                          choices=['bf', 'pot_dtaad', 'pot_evt', 'pot_dynamic',
                                   'both', 'all'],
                          help="bf=原有 oracle 阈值搜索（默认）；both=bf+pot_dtaad；all=bf+三种 POT")
        self.add_argument('--pot-q', type=float, default=DTAAD_Q, help='SPOT 风险水平')
        self.add_argument('--pot-level', type=float, default=None,
                          help='初始阈值分位数；None 时按 DTAAD 的 lm_d 表取')
        self.add_argument('--pot-scale', type=float, default=None,
                          help='阈值放大系数；None 时按 DTAAD 的 lm_d 表取')
        self.add_argument('--pot-init-n', type=int, default=-1,
                          help='init_score 取训练集前多少个窗口；<=0 表示用**全部**训练窗口，'
                               '与 DTAAD 的 pot_eval(lossTfinal, ...)（全程训练集）一致')
        self.add_argument('--pot-fixed', action='store_true',
                          help='用全数据集固定参数 (q=1e-3, level=0.98, scale=1) 作为对照列')
        self.parse_args(namespace=self)
        # 各数据集的默认值：只在命令行**没有**显式给出时才套用
        if self.window_length is None:
            self.window_length = 35 if self.dataset == 'SWaT' else 70
        if self.batch_size is None:
            self.batch_size = 24 if self.dataset == 'WADI' else 60
            # self.window_length = 140
        # elif self.dataset =='MSL':
        #     # self.window_length = 70
        #     self.latent_size=256 #?256？512貌981226
        # 似就不好了？
        #*下面是为了超参来设置的
        

def add_noise(data, noise_ratio, kind='mixed'):
    # 计算要替换的噪声数量
    num_samples = int(data.shape[0] * noise_ratio)
    if num_samples <= 0:
        return data
    # 随机选择要替换的索引
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    # 生成随机噪声
    # 基础版：整行替换为标准正态 N(0,1)
    if kind == 'gauss':
        data[indices] = np.random.normal(0, 1, data[indices].shape)
        return data
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




# 阈值模式 -> 要计算的 POT 口径（'bf' 不跑 POT）
POT_MODE_MAP = {
    'bf':          (),
    'pot_dtaad':   ('pot_dtaad',),
    'pot_evt':     ('pot_evt',),
    'pot_dynamic': ('pot_dynamic',),
    'both':        ('pot_dtaad',),
    'all':         ('pot_dtaad', 'pot_evt', 'pot_dynamic'),
}


def compute_init_score(model, train_data, config, n):
    """POT 的 init_score：训练集窗口的分数（训练集无标签）。

    n <= 0 时用**全部**训练窗口，与 DTAAD 的 `pot_eval(lossTfinal, ...)` 一致
    （DTAAD 对整段 trainD 前向取 lossTfinal）。n > 0 时只取前 n 个窗口。

    单独建一个 shuffle=False 的 loader，保证可复现；不使用 train_loader
    （后者是 shuffle=True 的，且只应服务于训练）。
    """
    if n is None or n <= 0 or n >= train_data.shape[0]:
        sub = train_data
    else:
        sub = train_data[:n]
    num_i = int(sub.shape[0] / config.batch_size)
    con_i = sub.shape[0] % config.batch_size
    loader = get_loader(sub, batch_size=config.batch_size,
                        window_length=config.window_length,
                        input_size=config.input_dim, shuffle=False)
    s = model.is_anomaly(loader, num_i, con_i)
    return s.detach().cpu().numpy() if hasattr(s, 'detach') else np.asarray(s)


def main():
    config = ExpConfig()
    # args = argparse.ArgumentParser()
    # args = vars(args)

    set_seed(config.seed)
    config.input_dim= get_data_dim(config.dataset)
    print('==============================================================================================')
    print('name:',config.name)
    print('backbone:',config.backbone)
    print('seed:',config.seed)
    print('batch:',config.batch_size)
    print('window_length',config.window_length)
    # print('hidden_size',config.hidden_size)
    print('latent_size',config.latent_size)
    print('trian or test:',config.trainmode)
    print("noise:",config.noise)

    (train_data, _), (test_data, test_label) = \
        get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
                 test_start=config.test_start)

    contam = None
    if config.contam_file:            # PRD4text：训练集由 bench/gen_contam.py 物化，覆盖上面载入的 clean
        # ⚠️ 该 .npy 已经过 preprocess（scaler 拟合于 clean），**不得再 preprocess 一次**
        #    否则会二次归一化，把注入的异常压掉（见 gen_contam.py 顶部说明 1）
        train_data = np.load(config.contam_file).astype(np.float32)
        if train_data.shape[1] != config.input_dim:
            raise SystemExit(f'[FAIL] 污染文件维度 {train_data.shape[1]} != '
                             f'数据集 {config.dataset} 的 {config.input_dim}')
        if config.noise:
            print('[warn] 指定了 --contam-file，--noise 被忽略（污染已物化在文件里）')
        man_path = config.contam_manifest
        if man_path is None:
            man_path = os.path.splitext(config.contam_file)[0] + '.json'
        if os.path.exists(man_path):
            with open(man_path) as f:
                contam = json.load(f)
            print(f"contam: rho={contam['rho']}  rho_win={contam['rho_win']:.4f}  "
                  f"n_contaminated={contam['n_contaminated']}  "
                  f"n_segments={contam['n_segments']}  alpha={contam['alpha']}")
        else:
            print(f'[warn] 找不到 manifest {man_path}，contam 字段留空')
    else:
        # 在加载数据后添加噪声（旧路径，保留不动）
        train_data = add_noise(train_data, config.noise, config.noise_kind)

    # ── 路 B：测试侧扰动（干净训练 → 含异常检测）─────────────────────────
    # 训练侧不动，把物化的污染数组替换 test_data；test_label 保持不变（标签仍是真的）。
    # 与 --contam-file 同理：该 .npy 已 preprocess 过，**不得再 preprocess**。
    # 注意 val 是从 test 尾部切出来的（main.py 下方的 val⊂test 既存缺陷），
    # 所以替换 test_data 会同时影响 val 与阈值选择 —— 这正是"部署时输入一直有扰"的语义。
    contam_test = None
    if config.contam_test_file:
        te = np.load(config.contam_test_file).astype(np.float32)
        if te.shape != test_data.shape:
            raise SystemExit(f'[FAIL] 测试污染文件形状 {te.shape} != test_data {test_data.shape}')
        _ndiff = int((te != test_data).any(1).sum())
        print(f'[dbg] contam_test vs 干净 test_data: 不同行={_ndiff}  '
              f'te.sum={float(te.sum()):.6f} clean.sum={float(test_data.sum()):.6f}')
        if _ndiff == 0:
            raise SystemExit('[FAIL] 污染文件与干净测试集逐位相同 —— 注入没生效')
        test_data = te
        man_path = config.contam_manifest or \
            (os.path.splitext(config.contam_test_file)[0] + '.json')
        if os.path.exists(man_path):
            with open(man_path) as f:
                contam_test = json.load(f)
            print(f"contam_test: rho={contam_test['rho']}  "
                  f"rho_win={contam_test['rho_win']:.4f}  "
                  f"n_contaminated={contam_test['n_contaminated']}  "
                  f"n_segments={contam_test['n_segments']}  alpha={contam_test['alpha']}")
        else:
            print(f'[warn] 找不到 test manifest {man_path}')

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
    import importlib
    _mod = importlib.import_module(BACKBONE_REGISTRY[config.backbone])
    model = _mod.patchattmamba(config.input_dim, w_size, config.latent_size, config.batch_size, config.window_length, config.out_dim,args=argsa)
    ##?因为这个导致的问题？？
    model = model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    n_midstack = sum(p.numel() for p in model.attmamba.parameters())
    print(f'params={n_params/1e6:.4f}M  中间块={n_midstack/1e6:.4f}M ({100*n_midstack/n_params:.2f}%)')

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 注意：这里原本有一句 `import os`。函数内的 import 会把 `os` 变成**整个函数的局部名**，
    # 于是函数前面（污染文件路径处理）对 os 的使用会 UnboundLocalError。模块顶部已 import os。
    # ckpt 路径：--ckpt-dir 优先。并行的同变体 run 必须各给一个，否则并发写同一个 best.pt。
    checkpoint = config.ckpt_dir or os.path.join('checkpoint', f'{config.name}/{config.dataset}')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    save_path = os.path.join(checkpoint,'best.pt')
    
    train_timing = None   # 每 epoch 真实耗时（供 PRD4text_r2 §12.2 用实测替换估算）
    if 'train' in config.trainmode:
        flag = 0
        f1 = -1
        epoch_secs = []
        for epoch in tqdm(range(10)):
            t_epoch = time.time()
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
            epoch_secs.append(time.time() - t_epoch)
            print(f'[timing] epoch {epoch}: {epoch_secs[-1]:.1f}s '
                  f'(steps={i})', flush=True)
        train_timing = {
            'epoch_secs': epoch_secs,
            'mean_epoch_sec': sum(epoch_secs) / len(epoch_secs),
            'n_windows': int(train_data.shape[0]),
            'n_steps_per_epoch': int(len(train_loader)),
            'batch_size': config.batch_size,
        }
        print(f'[timing] mean {train_timing["mean_epoch_sec"]:.1f}s/epoch  '
              f'windows={train_timing["n_windows"]}', flush=True)

    if 'test' in config.trainmode:
        if config.loadpath == 'None':#？在参数中写入的是str结构，所以这里。
            config.loadpath = save_path
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

        if config.score_dump:      # PRD5.2：评测协议不变，额外存分数以便事后算未调整 AUPRC
            _d = os.path.dirname(config.score_dump)
            if _d:
                os.makedirs(_d, exist_ok=True)
            _ts = np.asarray(test_score.detach().cpu().numpy()).ravel()
            import numpy as _np
            _np.savez_compressed(
                config.score_dump,
                test_score=_ts,
                test_label=_np.asarray(test_label[-len(_ts):]).ravel(),
                bf_threshold=float(th), bf_f1=float(t[0]))
            print(f'[score-dump] {config.score_dump}  n={_ts.size}  '
                  f'AUPRC(未调整) 待离线算')

        #* ---- POT 阈值（标签无关）。threshold-mode='bf' 时这一段完全不执行 ----
        if POT_MODE_MAP[config.threshold_mode]:
            if config.pot_fixed:
                paramset, q = 'fixed', 1e-3
                level, scale = 0.98, 1.0
            else:
                paramset, q = 'dtaad', config.pot_q
                lm = DTAAD_LM.get(config.dataset, (0.98, 1.0))
                level = lm[0] if config.pot_level is None else config.pot_level
                scale = lm[1] if config.pot_scale is None else config.pot_scale
            modes = POT_MODE_MAP[config.threshold_mode]
            print(f'--- POT (paramset={paramset}, q={q}, level={level}, scale={scale}) ---')
            init_score = compute_init_score(model, train_data, config, config.pot_init_n)
            pres = pot_eval(init_score, test_score.detach().cpu().numpy(),
                            test_label[-len(test_score):],
                            q=q, level=level, scale=scale, modes=modes)
            print(f"  init_score n={pres['init_score_size']}  fit_ok={pres['fit_ok']}  "
                  f"init_threshold={pres.get('init_threshold')}  "
                  f"真EVT={pres.get('extreme_quantile_raw')}")
            for m, r in pres['modes'].items():
                mt = r['metrics']
                print(f"  {m:12s} th={r['threshold']:.6f}  F1={mt[0]:.4f}  "
                      f"P={mt[1]:.4f}  R={mt[2]:.4f}")

            out_dir2 = os.path.join('results', 'threshold')
            os.makedirs(out_dir2, exist_ok=True)
            out_json2 = os.path.join(out_dir2, f"SECOND_{config.dataset}_{paramset}.json")
            with open(out_json2, 'w') as f:
                json.dump({
                    'model': 'SECOND', 'dataset': config.dataset,
                    'threshold_mode': config.threshold_mode,
                    'config': {'input_dim': config.input_dim, 'out_dim': config.out_dim,
                               'window_length': config.window_length,
                               'latent_size': config.latent_size,
                               'batch_size': config.batch_size,
                               'n_layer': argsa.n_layer},
                    'bf_search_oracle': {'f1': float(t[0]), 'precision': float(t[1]),
                                         'recall': float(t[2]), 'threshold_pct': float(th)},
                    'pot': {'paramset': paramset, 'q': q, 'level': level, 'scale': scale,
                            'modes': modes, 'fit_ok': pres['fit_ok'],
                            'init_score': {'source': 'train_first_n',
                                           'n': pres['init_score_size'],
                                           'n_requested': config.pot_init_n},
                            'init_threshold': pres.get('init_threshold'),
                            'extreme_quantile_raw': pres.get('extreme_quantile_raw'),
                            'level_used': pres.get('level_used'),
                            'results': {m: {'threshold': r['threshold'],
                                            'f1': r['metrics'][0], 'precision': r['metrics'][1],
                                            'recall': r['metrics'][2],
                                            'TP': int(r['metrics'][3]), 'TN': int(r['metrics'][4]),
                                            'FP': int(r['metrics'][5]), 'FN': int(r['metrics'][6])}
                                        for m, r in pres['modes'].items()}},
                    'ckpt': {'path': os.path.abspath(config.loadpath)},
                    'env': {'gpu': torch.cuda.get_device_name(0),
                            'torch': torch.__version__, 'cuda': torch.version.cuda},
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }, f, indent=2, ensure_ascii=False)
            print('POT saved to:', out_json2)

        #* 结果落盘（与 MUTANT 的 schema 对齐，便于统一聚合）
        out_dir = config.result_dir
        os.makedirs(out_dir, exist_ok=True)
        rec = {
            'model': 'SECOND',
            'backbone': config.backbone,
            'dataset': config.dataset,
            'seed': config.seed,
            'config': {'input_dim': config.input_dim, 'out_dim': config.out_dim,
                       'window_length': config.window_length, 'latent_size': config.latent_size,
                       'batch_size': config.batch_size, 'n_layer': argsa.n_layer,
                       'epochs': 10, 'lr': 0.01, 'N': config.N, 'val': config.val,
                       'noise': config.noise,
                       'noise_kind': config.noise_kind},
            'params': {'n_params': n_params, 'n_params_M': n_params / 1e6,
                       'n_midstack': n_midstack, 'n_midstack_M': n_midstack / 1e6,
                       'midstack_share': n_midstack / n_params},
            'timing': train_timing,
            'contam': contam,
            'contam_test': contam_test,
            'metrics': {'f1': float(t[0]), 'precision': float(t[1]), 'recall': float(t[2]),
                        'threshold': float(th), 'TP': int(t[3]), 'TN': int(t[4]),
                        'FP': int(t[5]), 'FN': int(t[6])},
            'ckpt': {'path': os.path.abspath(config.loadpath)},
            'env': {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                    'torch': torch.__version__},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        out_name = config.result_name or \
            f"{config.backbone}_{config.dataset}_seed{config.seed}"
        out_json = os.path.join(out_dir, f"{out_name}.json")
        with open(out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print('metrics saved to:', out_json)

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

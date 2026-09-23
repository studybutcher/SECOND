import numpy as np

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label, threshold, pred=None,
                    calc_latency=False, abs_threshold=None):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
            **注意：这是「百分位」而不是绝对分数**（下面用 np.percentile 换算）。
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
        abs_threshold (float or None): 若给出，则**直接当作绝对阈值**使用，忽略 `threshold`。
            供 POT 等返回绝对阈值的策略使用。默认 None 时行为与改动前完全一致。

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if abs_threshold is not None:
        thresh = float(abs_threshold)      # POT：直接给绝对阈值
    else:
        thresh = np.percentile(score, float(threshold))#!不是用的原始值他妈的，是用的百分比。并且是在所有的这些得分中选出的一个百分位数，而不是直接乘比例。
    if pred is None:
        predict = score > thresh#*直接生成对应位置的？？？
        # for i,n in enumerate(predict):
        #     if n:
        #         print("哟！", score[i] ,thresh)
    else:
        predict = pred
    actual = label > 0.1#*相当于有值。一般也确实都为1或0
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:#*当突然出现异常的时候会标记
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:#*只有当标签正常的时候就设置状态为正常？？
            anomaly_state = False
        if anomaly_state:#*异常的持续性
            predict[i] = True

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t,predict
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label),predict


def bf_search(score, label, step_num):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    # threshold = search_lower_bound
    threshold = 93
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(step_num):
        # threshold += search_range / float(search_step)
        target ,predict= calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_predic = predict
        # if verbose and i % display_freq == 0:
        #     print("cur thr: ", threshold, target, m, m_t)
        threshold += 0.01#*动态选择阈值
    # print(m, m_t)
    return m, m_t,predict


# ============================================================================
# POT / SPOT 阈值选择（标签无关）
# ----------------------------------------------------------------------------
# 参考实现：`DTAAD-main/src/pot.py` 的 pot_eval + `DTAAD-main/src/spot.py`（SPOT 库）。
# 本仓库的 spot.py 为整文件拷入 `third_party/spot.py`。
#
# ⚠️ 重要发现（见 docs/PRD3_阈值选择策略.md §2.1.1）：
#   SPOT 的 run(dynamic=False) 在遇到第一个超过 init_threshold 的点之后，会把
#   extreme_quantile **永久**改写成 init_threshold（spot.py:417-420），因此
#   mean(thresholds) ≈ quantile(init_score, level)，GPD 拟合出的真 EVT 分位点被丢弃。
#   故 DTAAD 的有效阈值实为「训练分数的 level 分位数 × scale」。
#   本模块把三种口径都暴露出来，互不替代。
# ============================================================================

# DTAAD-main/src/constants.py 的 lm_d 表，取**实际生效**的那一项。
# 原代码 `1 if 'TranAD' or 'DTTAD' in args.model else 0` 中 'TranAD' 恒为真，
# 条件永远成立，故实际生效的是 index 1。
DTAAD_LM = {
    'SMAP': (0.98, 1.0),
    'MSL':  (0.999, 1.04),
    'SWaT': (0.993, 1.0),
    'WADI': (0.999, 1.0),
}
DTAAD_Q = 1e-5


def calc_seq_abs(score, label, abs_threshold):
    """以**绝对阈值**为界算 F1；point-adjust 逻辑与 calc_seq 一致。

    Returns:
        (list, np.ndarray): (f1, P, R, TP, TN, FP, FN), predict
    """
    predict, _ = adjust_predicts(score, label, None, calc_latency=True,
                                 abs_threshold=abs_threshold)
    return list(calc_point2point(predict, label)), predict


def _spot_initialize(init_score, score, q, level, max_try=1000):
    """建 SPOT → fit → initialize。GPD 拟合失败时按 DTAAD 的做法递减 level 重试。"""
    from third_party.spot import SPOT      # 延迟导入：避免 eval_method 被 import 时就拉 matplotlib
    lv, last_err = level, None
    for _ in range(max_try):
        try:
            sp = SPOT(q)
            sp.fit(init_score, score)
            sp.initialize(level=lv, min_extrema=False, verbose=False)
            return sp, lv, None
        except Exception as e:             # noqa: BLE001  拟合失败属预期情况
            last_err = e
            lv = lv * 0.999
    return None, lv, last_err


def pot_eval(init_score, score, label, q=DTAAD_Q, level=0.98, scale=1.0,
             modes=('pot_dtaad',)):
    """POT 阈值选择。**全程只用分数，label 仅用于算完阈值后的评测。**

    Args:
        init_score: 无标签的参照分数（本仓库用训练集前 N 个窗口的分数）
        score:      测试集分数
        label:      测试集标签（不参与阈值计算）
        q / level / scale: SPOT 参数与 DTAAD 的放大系数
        modes:      要算的口径
            'pot_dtaad'   → mean(run(dynamic=False)['thresholds']) * scale
                            ≈ quantile(init_score, level) * scale（复现 DTAAD 的有效行为）
            'pot_evt'     → initialize() 之后的 extreme_quantile * scale（真 EVT 分位点）
            'pot_dynamic' → mean(run(dynamic=True)['thresholds']) * scale（在线自适应）

    Returns:
        dict: 含 'modes': {mode: {'metrics': 7-tuple, 'threshold': float, 'predict': ndarray}}
              以及 init_threshold / extreme_quantile_raw / level_used / fit_ok
    """
    init_score = np.asarray(init_score, dtype=float).ravel()
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label).ravel()
    if len(score) != len(label):
        raise ValueError('score and label must have the same length')

    out = {'init_score_size': int(init_score.size), 'level_requested': float(level),
           'q': q, 'scale': scale, 'fit_ok': True, 'modes': {}}

    sp_evt, lv, err = _spot_initialize(init_score, score, q, level)
    if sp_evt is None:
        # GPD 拟合彻底失败 → 回退到「训练分数的 level 分位数」，并如实标注
        out['fit_ok'] = False
        out['fit_error'] = repr(err)[:200]
        out['level_used'] = float(level)
        th = float(np.quantile(init_score, level)) * scale
        for m in modes:
            met, pred = calc_seq_abs(score, label, th)
            out['modes'][m] = {'threshold': th, 'metrics': met, 'predict': pred,
                               'fallback': True}
        return out

    out['level_used'] = float(lv)
    out['init_threshold'] = float(sp_evt.init_threshold)
    out['extreme_quantile_raw'] = float(sp_evt.extreme_quantile)

    for m in modes:
        if m == 'pot_evt':
            th = float(sp_evt.extreme_quantile) * scale      # 必须在 run() 之前取
        else:
            # run() 会改写 extreme_quantile，故每种口径各建一个干净实例
            sp, _, _ = _spot_initialize(init_score, score, q, level)
            th = float(np.mean(sp.run(dynamic=(m == 'pot_dynamic'))['thresholds'])) * scale
        met, pred = calc_seq_abs(score, label, th)
        out['modes'][m] = {'threshold': th, 'metrics': met, 'predict': pred}
    return out
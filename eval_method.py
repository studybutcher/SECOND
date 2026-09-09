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
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
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
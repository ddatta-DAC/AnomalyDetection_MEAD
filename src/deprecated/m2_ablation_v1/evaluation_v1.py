import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
import matplotlib.pyplot as plt


def precision_recall_curve(
        sorted_id_score_dict,
        anomaly_id_list
):
    recall = 0
    correct = 0
    recall_vals = []
    precision_vals = []

    # True anomalies
    num_anomalies = len(anomaly_id_list)
    print('Number of Anomalies (true anomaly cases)', num_anomalies)
    total_count = len(sorted_id_score_dict)
    print(total_count)
    input_ids = list(sorted_id_score_dict.keys())


    '''
    Following Charu Aggarwal
    Precision  = (S(t) intersection G) /  |S(t)|
    Recall  = (S(t) intersection G) /  |G|
    '''

    print('------')

    step = 0.125
    _k_prev = None
    for t in np.arange(0.25, 100 + 0.125, step):
        _k = int((t/100)*total_count)
        if _k == 0 : continue
        if _k == _k_prev :
            continue
        _numerator = len(set(input_ids[:_k]).intersection(anomaly_id_list))

        p = _numerator/_k
        r = _numerator/num_anomalies

        precision_vals.append(p)
        recall_vals.append(r)
        if r == 1 :
            break
        _k_prev = _k
        print(p,r)
        # -------------------------- #

    return recall_vals, precision_vals

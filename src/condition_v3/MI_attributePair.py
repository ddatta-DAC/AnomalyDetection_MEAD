import pickle
from sklearn.metrics import mutual_info_score
import itertools
import time
import sys
import pandas as pd
import numpy as np
import math

def get_MI_attrSetPair(
        data_x,
        s1,
        s2,
        obj_adtree
):
    if len(s1) == 1 and len(s2) == 1:
        _x = np.reshape(data_x[:, s1], -1)
        _y = np.reshape(data_x[:, s2], -1)
        return  mutual_info_score(_x, _y)

    def get_count(row, obj_adtree , cols):
        dict_dom_vals = { c : row[c] for c in cols }
        return  obj_adtree.get_count(dict_dom_vals)

    num_rows = data_x.shape[0]

    _df_s1 = pd.DataFrame(data_x[:, s1])
    tmp = { i: s1[i] for i in range(len(s1))}
    _df_s1 = _df_s1.rename(columns = tmp)
    _df_s1.drop_duplicates(inplace=True)
    cols = list(s1)
    _df_s1['px'] = _df_s1.apply(get_count,axis=1, args=(obj_adtree , cols))
    _df_s1['px'] = _df_s1['px'] / num_rows

    _df_s2 = pd.DataFrame(data_x[:, s2])
    tmp = {i: s2[i] for i in range(len(s2))}
    _df_s2 = _df_s2.rename(columns=tmp)
    _df_s2.drop_duplicates(inplace=True)
    cols = list(s2)
    _df_s2['py'] = _df_s2.apply(get_count, axis=1, args=(obj_adtree, cols))
    _df_s2['py'] = _df_s2['py']/num_rows

    s1_s2 = list(s1)
    s1_s2.extend(s2)
    df_0 = pd.DataFrame(
        data_x[:, s1_s2]
    )
    tmp = {i: s1_s2[i] for i in range(len(s1_s2))}
    df_0 = df_0.rename(columns=tmp)
    df_0.drop_duplicates(inplace=True)
    cols = list(s1_s2)
    df_0['pxy'] = df_0.apply(get_count, axis=1, args=(obj_adtree, cols))
    df_0['pxy'] =  df_0['pxy'] /num_rows

    df_0 = df_0.join(_df_s1.set_index(s1), on= s1, how= 'outer')
    df_0 = df_0.join(_df_s2.set_index(s2), on= s2, how= 'outer')
    _eps = 0.00000001
    df_0['px'] = df_0['px'] + _eps
    df_0['py'] = df_0['py'] + _eps
    df_0['pxy'] = df_0['pxy'] + _eps

    def sum_terms(row):
        res = row['pxy'] *  math.log(row['pxy']/ (row['px'] * row['py']),2 )
        return res

    df_0['_mi'] = df_0.apply(sum_terms,axis=1)
    _mi = sum(list(df_0['_mi']))
    return _mi

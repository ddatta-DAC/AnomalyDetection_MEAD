import numpy as np
import yaml
import pandas as pd
import sklearn
from pprint import pprint
import glob
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import math
from sklearn import preprocessing
from scipy.stats import rv_discrete
import pickle
from sklearn.metrics import mutual_info_score
import itertools
import time
import sys
import operator
import logging
import logging.handlers

sys.path.append('./..')
sys.path.append('./../..')
from collections import OrderedDict
from joblib import parallel_backend
from joblib import Parallel, delayed

try:
    from src.Eval import eval_v1 as eval
except:
    from .src.Eval import eval_v1 as eval

try:
    from src.data_fetcher import data_fetcher
except:
    from .src.data_fetcher import data_fetcher

try:
    import ad_tree_v1
except:
    from . import ad_tree_v1

# ------------------------- #
# Based on
# Detecting patterns of anomalies
# https://dl.acm.org/citation.cfm?id=1714140
# ------------------------- #
_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "7.0"
# ------------------------- #

_DIR = None
DATA_DIR = None
CONFIG_FILE = 'config_1.yaml'
ID_LIST = None
SAVE_DIR = None
OP_DIR = None
config = None
# Algorithm thresholds
MI_THRESHOLD = 0.1
ALPHA = 0.1
DISCARD_0 = True
logger = None

# ---------------------------------- #

def get_data(data_dir, dir):
    train_x_pos, _, _, _, domain_dims = data_fetcher.get_data_v3(
        data_dir,
        dir,
        c=1
    )
    test_dict_cIdx_data = { }
    for c in range(1,3+1):
        _, _, test_pos, test_anomaly, _ = data_fetcher.get_data_v3(
            data_dir,
            dir,
            c=c
        )

        test_pos_idList = test_pos[0]
        test_pos_x = test_pos[1]
        test_anomaly_idList = test_anomaly[0]
        test_anomaly_x = test_anomaly[1]

        test_ids = list(np.hstack(
            [test_pos_idList,
             test_anomaly_idList]
        ))

        test_data_x = np.vstack([
            test_pos_x,
            test_anomaly_x
        ])

        test_dict_cIdx_data[c] = [test_ids, test_data_x, test_anomaly_idList]

    return train_x_pos, test_dict_cIdx_data


# ------------------------------ #

def setup(_dir=None):
    global CONFIG_FILE
    global _DIR
    global DATA_DIR
    global ID_LIST
    global OP_DIR
    global _DIR
    global SAVE_DIR
    global config
    global use_mi
    global logger
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    SAVE_DIR = config['SAVE_DIR']
    if _dir is None:
        _DIR = config['_DIR']
    else:
        _DIR = _dir
    OP_DIR = config['OP_DIR']
    DATA_DIR = config['DATA_DIR']
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    SAVE_DIR = os.path.join(SAVE_DIR, _DIR)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)
    OP_DIR = os.path.join(OP_DIR, _DIR)

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)
    use_mi = True

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    log_file = config[_DIR]['log_file']
    handler = logging.FileHandler(os.path.join(OP_DIR, log_file))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(' Info start ')



# ----------------------------------- #
def calc_MI(x, y):
    mi = mutual_info_score(x, y)
    return mi


# ----------------------------------- #

# Get arity of each domain
def get_domain_arity():
    global DATA_DIR
    global _DIR
    f = os.path.join(DATA_DIR, _DIR, 'domain_dims.pkl')
    with open(f, 'rb') as fh:
        dd = pickle.load(fh)
    return list(dd.values())


# ----------------------------------- #

def get_MI_attrSetPair(
        data_x,
        s1,
        s2,
        obj_adtree
):
    if len(s1) > 1 or len(s2) > 1:
        return 1

    if len(s1) == 1 or len(s2) == 1:
        _x = np.reshape(data_x[:, s1], -1)
        _y = np.reshape(data_x[:, s2], -1)
        return calc_MI(x=_x, y=_y)

    def _join(row, indices):
        r = '_'.join([str(row[i]) for i in indices])
        return r

    mask = np.random.choice([False, True], len(data_x), p=[0.8, 0.2])
    data_x = data_x[mask]

    _idx = list(s1)
    _idx.extend(s2)
    _atr = list(s1)
    _atr.extend(s2)
    _dict = {}

    for a in _atr:
        _dict[a] = set(data_x[:, [a]])

    _tmp_df = pd.DataFrame(
        data=data_x,
        copy=True
    )
    _tmp_df = _tmp_df[_atr]

    _tmp_df['x'] = None
    _tmp_df['y'] = None
    _tmp_df['x'] = _tmp_df.apply(
        _join,
        axis=1,
        args=(s1,)
    )
    _tmp_df['y'] = _tmp_df.apply(
        _join,
        axis=1,
        args=(s2,)
    )
    mi = calc_MI(_tmp_df['x'], _tmp_df['y'])
    return mi


# MI = Sum ( P_(x)(y) log( P_(x)(y)/ P_(x)*P_(y) )

'''
get sets of attributes for computing r-value
input attribute indices 0 ... m-1
Returns sets of attributes of size k
'''


def get_attribute_sets(
        data_x,
        attribute_list,
        obj_adtree,
        k=1
):
    global SAVE_DIR
    global use_mi

    # check if file present in save dir
    op_file_name = 'set_pairs_' + str(k) + '.pkl'
    op_file_path = os.path.join(SAVE_DIR, op_file_name)

    if os.path.exists(op_file_path):
        with open(op_file_path, 'rb') as fh:
            set_pairs = pickle.load(fh)
        return set_pairs

    # -------------------------------------- #

    # We can attribute sets till size k
    # Add in size 1
    sets = list(
        itertools.combinations(attribute_list, 1)
    )

    for _k in range(2, k + 1):
        _tmp = list(itertools.combinations(attribute_list, _k))
        sets.extend(_tmp)

    '''
    Check if 2 sets have MI > 0.1 and are mutually exclusive
    TODO : implement concurrency
    '''

    set_pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            s1 = sets[i]
            s2 = sets[j]

            # Mutual exclusivity test
            m_e = (len(set(s1).intersection(s2)) == 0)
            mi_flag = False
            if m_e is False:
                continue
            # -- Ignore MI for now -- #
            # MI
            if use_mi is False:
                mi_flag = True
            else:
                mi = get_MI_attrSetPair(
                    data_x,
                    s1,
                    s2,
                    obj_adtree
                )
                if mi >= 0.1:
                    mi_flag = True

            if mi_flag is True:
                set_pairs.append((s1, s2))

    _dict = {
        e[0]: e[1] for e in enumerate(set_pairs, 0)
    }
    set_pairs = _dict

    # Save the result
    with open(op_file_path, 'wb') as fh:
        pickle.dump(set_pairs, fh, pickle.HIGHEST_PROTOCOL)

    return set_pairs


def get_count(obj_adtree, domains, vals):
    _dict = {
        k: v for k, v in zip(domains, vals)
    }
    res = obj_adtree.get_count(_dict)
    return res


def get_r_value(
        _id,
        record,
        obj_adtree,
        set_pairs,
        N
    ):
    global ALPHA
    global DISCARD_0
    _r_dict = {}

    for set_pair_idx, set_pair_item in set_pairs.items():

        _vals = []
        _domains = []
        _vals_1 = []
        _domains_1 = []
        _vals_2 = []
        _domains_2 = []

        for _d in set_pair_item[0]:
            _domains_1.append(_d)
            _domains.append(_d)
            _vals_1.append(record[_d])
            _vals.append(record[_d])
        for _d in set_pair_item[1]:
            _domains_2.append(_d)
            _domains.append(_d)
            _vals_2.append(record[_d])
            _vals.append(record[_d])

        P_at = get_count(
            obj_adtree,
            _domains_1,
            _vals_1) + 1

        P_at = P_at / (N + 2)

        P_bt = get_count(
            obj_adtree,
            _domains_2,
            _vals_2
        ) + 1
        P_bt = P_bt / (N + 2)

        P_ab = get_count(obj_adtree, _domains, _vals) / N
        r = (P_ab) / (P_at * P_bt)
        _r_dict[set_pair_idx] = r

    sorted_r = list(
        sorted(
            _r_dict.items(),
            key=operator.itemgetter(1)
        )
    )

    score = 1
    U = set()
    threshold = ALPHA

    for i in range(len(sorted_r)):
        _r = sorted_r[i][1]
        tmp = set_pairs[sorted_r[i][0]]
        _attr = [item for sublist in tmp for item in sublist]

        if _r > threshold:
            break
        if DISCARD_0 and _r <= 0.0:
            continue

        if len(U.intersection(set(_attr))) == 0:
            U = U.union(set(_attr))
            score *= _r

    return _id, score


def process(_dir=None):
    global DATA_DIR
    global _DIR
    global config
    global OP_DIR
    global logger

    setup(_dir)

    logger.info('--------')
    logger.info(_DIR)
    logger.info('--------')

    k_val = None

    train_x, test_dict_cIdx_data = get_data(DATA_DIR, _DIR)

    if k_val is not None:
        K = k_val
    else:
        K = int(config['K'])

    ''' Start of train '''
    # Number of training instances
    N = train_x.shape[0]
    obj_ADTree = ad_tree_v1.ADT()
    obj_ADTree.setup(train_x)

    attribute_list = list(
        range(train_x.shape[1])
    )

    attribute_set_pairs = get_attribute_sets(
        train_x,
        attribute_list,
        obj_ADTree,
        k=K
    )

    print(' Number of attribute set pairs ', len(attribute_set_pairs))

    ''' Start of test '''
    for c, _t_data in  test_dict_cIdx_data.items():

        test_ids = _t_data[0]
        test_data_x = _t_data[1]
        test_anomaly_idList = _t_data[2]

        start = time.time()

        results = OrderedDict()
        for _id, record in zip(test_ids, test_data_x):
            _, r_score = get_r_value(
                _id,
                record,
                obj_ADTree,
                attribute_set_pairs,
                N
            )
            results[_id] = r_score



        end = time.time()
        _time = end - start

        print(' Time taken :',  end - start)

        tmp = sorted(results.items(), key=operator.itemgetter(1))
        sorted_id_score_dict = OrderedDict()
        for e in tmp:
            sorted_id_score_dict[e[0]] = e[1]



        recall, precison = eval.precision_recall_curve(
            sorted_id_score_dict = sorted_id_score_dict,
            anomaly_id_list = test_anomaly_idList
        )

        _auc = auc(recall, precison)
        print('AUC ', _auc)
        logger.info('c = ' + str(c))
        logger.info('Time taken ' + str(_time))
        logger.info('AUC : ' + str(_auc))

        plt.figure(figsize=[14, 8])
        plt.plot(
            recall,
            precison,
            color='blue', linewidth=1.75)
        plt.xlabel('Recall', fontsize=15)
        plt.ylabel('Precision', fontsize=15)
        plt.title('Precision Recall | AUC ' + str(_auc), fontsize=15)
        f_name = 'precison-recall_1'+ '_test_' + str(c) + '.png'
        f_path = os.path.join(OP_DIR, f_name)
        plt.savefig(f_path)
        plt.close()


# ------------------------------------------ #


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", nargs='?', default="None")
args = parser.parse_args()

if args.dir == 'None':
    _dir = None
else:
    _dir = args.dir

process(_dir)
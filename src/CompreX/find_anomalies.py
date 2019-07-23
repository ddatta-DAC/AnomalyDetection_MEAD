import matplotlib.pyplot as plt
import os
import time
import glob
from collections import OrderedDict
import numpy as np
import pandas as pd
import logging
import logging.handlers
import yaml
import pickle
import inspect
import sys
import operator
from sklearn.metrics import auc

sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../.')

cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-2]
)
sys.path.append(cur_path)

try:
    from src.Eval import eval_v1 as eval
except:
    from .src.Eval import eval_v1 as eval

try:
    from src.CompreX.comprex.comprex import CompreX
except:
    from .src.CompreX.comprex.comprex import CompreX

try:
    from src.data_fetcher import data_fetcher
except:
    from .src.data_fetcher import data_fetcher

CONFIG_FILE = 'config_compreX.yaml'
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

MODEL_NAME = None
OP_DIR = None
DATA_DIR = None
SAVE_DIR = None
_DIR = None


# --------------------------- #

def trivial_test():
    X = pd.DataFrame([
        ['a', 'b', 'x'],
        ['a', 'b', 'x'],
        ['a', 'b', 'x'],
        ['a', 'b', 'x'],
        ['a', 'c', 'x'],
        ['a', 'c', 'y'],
        ['a', 'b', 'x']
    ],
        columns=['f1', 'f2', 'f3'],
        index=[i for i in np.arange(7, 14)],
        dtype='category'
    )

    estimator = CompreX(logging_level=logging.ERROR)
    estimator.transform(X)
    estimator.fit(X)
    res = estimator.predict(X)
    print(type(res))
    print(res)

# --------------------------- #

def setup():
    global SAVE_DIR
    global _DIR
    global config
    global OP_DIR
    global MODEL_NAME
    global DATA_DIR
    global domain_dims
    global cur_path

    SAVE_DIR = config['SAVE_DIR']
    _DIR = config['_DIR']
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
    MODEL_NAME = 'compreX'

    # if not os.path.exists(SAVE_DIR):
    #     os.mkdir(SAVE_DIR)
    #     if os.path.exists(os.path.join(SAVE_DIR, _DIR)):
    #         os.mkdir(os.path.join(SAVE_DIR, _DIR))



'''
Ensure the columns have different values.
Append column id to each value
'''

def get_data(data_dir, dir):

    def stringify_data(arr) -> np.array:
        tmp1 = []
        for i in range(arr.shape[0]):
            tmp2 = []
            for j in range(arr.shape[1]):
                tmp2.append(str(arr[i][j]) + '_' + str(j))
            tmp1.append(tmp2)

        tmp1 = np.array(tmp1)
        return tmp1

    train_x_pos, _, _, _, domain_dims = data_fetcher.get_data_v3(
        data_dir,
        dir,
        c=1
    )

    train_x_pos = stringify_data(train_x_pos)

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

        test_data_x = stringify_data(test_data_x)
        test_dict_cIdx_data[c] = [test_ids, test_data_x, test_anomaly_idList]

    return train_x_pos, test_dict_cIdx_data



# --------------------------- #
def main():
    global DATA_DIR
    global _TIME_IT
    global _DIR
    global OP_DIR
    global SAVE_DIR
    global config
    setup()

    train_x_pos, test_dict_cIdx_data = get_data(DATA_DIR, _DIR)

    '''
    TRAIN model
    '''
    # ---- Core ------ #
    _df_input = []
    for _j in range(train_x_pos.shape[0]):
        _df_input.append(list(train_x_pos[_j]))

    cols = ['f' + str(j) for j in range(train_x_pos.shape[1])]
    X = pd.DataFrame(
        _df_input,
        columns=cols,
        index=[_j for _j in range(train_x_pos.shape[0])],
        dtype='category'
    )

    estimator = CompreX(
        logging_level=logging.ERROR
    )
    estimator.transform(X)
    estimator.fit(X)

    ''' Start of test '''
    for c, _t_data in test_dict_cIdx_data.items():
        test_ids = _t_data[0]
        test_data_x = _t_data[1]
        test_anomaly_idList = _t_data[2]
        start = time.time()


        test_result_r = []
        test_result_p = []

        start_time = time.time()
        res = estimator.predict(test_data_x)
        '''
            'res' is ordered in the order of the input
            match it with the ordered list of ids
        '''
        anomaly_scores = list(res)
        anomaly_score_dict = {
            k:v
            for k,v in zip(test_ids,anomaly_scores)
        }

        # --------------- #
        ''' 
        Sort in reverse order, since higher score means anomaly 
        '''

        tmp = sorted(
            anomaly_score_dict.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        sorted_id_score_dict = OrderedDict()
        for e in tmp:
            sorted_id_score_dict[e[0]] = e[1]

        recall, precison = eval.precision_recall_curve(
            sorted_id_score_dict,
            anomaly_id_list=test_anomaly_idList
        )

        end_time = time.time()
        time_taken = end_time - start_time
        _auc = auc(recall, precison)

        print('Time taken [seconds]', time_taken , 'AUC',  _auc)
        print('--------------------------')



# ---------------------------- #

main()


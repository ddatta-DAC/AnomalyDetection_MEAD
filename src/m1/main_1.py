# --------------------------------- #
# Calculate  record embeddings based on Arora Paper
# --------------------------------- #
# this is to verify for both sets
# ----------------------------------- #

import operator
import pickle
import numpy as np
import glob
import json
import pandas as pd
import os
import argparse
import sys
import matplotlib
import time
from pprint import pprint
import inspect
from collections import OrderedDict
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import auc

# matplotlib.use('Agg')
sys.path.append('./..')
sys.path.append('./../../.')

import src.m1.lof_1 as lof_1

try:
    from .src.m1 import lof_1 as lof_1
except:
    from src.m1 import lof_1 as lof_1

try:
    from .src.m1 import tf_model_2 as tf_model
except:
    from src.m1 import tf_model_2 as tf_model

try:
    from .src.m1 import evaluation_v1 as evaluation_v1
except:
    from src.m1 import evaluation_v1 as evaluation_v1

try:
    from .src.m1 import isolation_forest as IF
except:
    from src.m1 import isolation_forest as IF

try:
    from .src.m1 import data_fetcher as data_fetcher
except:
    from src.m1 import data_fetcher as data_fetcher

# ------------------------------------ #

cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-1]
)

sys.path.append(cur_path)

_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "5.0"
__processor__ = 'embedding'
_SAVE_DIR = 'save_dir'
MODEL_NAME = None
_DIR = None
DATA_DIR = None
MODEL_OP_FILE_PATH = None
CONFIG_FILE = 'config_1.yaml'
CONFIG = None


# ----------------------------------------- #

def get_domain_dims():
    global DATA_DIR
    f_path = os.path.join(DATA_DIR, 'domain_dims.pkl')
    with open(f_path, 'rb') as fh:
        res = pickle.load(fh)
    return list(res.values())


# ----------------------------------------- #
# ---------               Model Config    --------- #
# ----------------------------------------- #

# embedding_dims = None
DOMAIN_DIMS = None


def setup_general_config():
    global MODEL_NAME
    global _DIR
    global SAVE_DIR
    global OP_DIR
    global _SAVE_DIR
    global CONFIG

    SAVE_DIR = os.path.join(CONFIG['SAVE_DIR'], _DIR)
    OP_DIR = os.path.join(CONFIG['OP_DIR'], _DIR)
    print(cur_path)
    print(OP_DIR)

    if not os.path.exists(CONFIG['OP_DIR']):
        os.mkdir(CONFIG['OP_DIR'])

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    if not os.path.exists(CONFIG['SAVE_DIR']):
        os.mkdir(os.path.join(CONFIG['SAVE_DIR']))

    if not os.path.exists(SAVE_DIR):
        os.mkdir(os.path.join(SAVE_DIR))
    return


# --------------------------------------------- #

def set_up_model(config, _dir):
    global embedding_dims
    global SAVE_DIR
    global OP_DIR
    global MODEL_NAME
    MODEL_NAME = config['MODEL_NAME']

    if type(config[_dir]['op_dims']) == str:
        embedding_dims = config[_dir]['op_dims']
        embedding_dims = embedding_dims.split(',')
        embedding_dims = [int(e) for e in embedding_dims]
    else:
        embedding_dims = config[_dir]['op_dims']

    model_obj = tf_model.model(MODEL_NAME, SAVE_DIR, OP_DIR)
    model_obj.set_model_options(
        show_loss_figure=config[_dir]['show_loss_figure'],
        save_loss_figure=config[_dir]['save_loss_figure']
    )

    domain_dims = get_domain_dims()
    LR = config[_dir]['learning_rate']
    model_obj.set_model_hyperparams(
        domain_dims=domain_dims,
        emb_dims=embedding_dims,
        batch_size=config[_dir]['batchsize'],
        num_epochs=config[_dir]['num_epochs'],
        learning_rate=LR,
        alpha=config[_dir]['alpha']
    )
    model_obj.inference = False
    model_obj.build_model()
    return model_obj


def process(
        CONFIG,
        _DIR,
        data_x,
        normal_ids,
        anomaly_ids
):
    model_obj = set_up_model(CONFIG, _DIR)

    return
    _use_pretrained = CONFIG[_DIR]['use_pretrained']

    if _use_pretrained is True:

        saved_file_path = None
        pretrained_file = CONFIG[_DIR]['saved_model_file']

        print('Pretrained File :', pretrained_file)
        saved_file_path = os.path.join(
            SAVE_DIR,
            'checkpoints',
            pretrained_file
        )
        if saved_file_path is not None:
            model_obj.set_pretrained_model_file(saved_file_path)
        else:
            model_obj.train_model(data_x)

    elif _use_pretrained is False:
        model_obj.train_model(data_x)


    score_op1 = model_obj.get_event_score()

    mean_embeddings = model_obj.get_embedding_mean(data_x)

    print('Number of true anomalies', len(anomaly_ids))

    # ---------------------
    # USE LOF here
    # ---------------------

    all_ids = list(anomaly_ids)
    all_ids.extend(normal_ids)

    sorted_id_score_dict = IF.anomaly_1(
        id_list=all_ids,
        embed_list=mean_embeddings
    )

    _scored_dict_test = OrderedDict(sorted_id_score_dict)
    recall, precison = evaluation_v1.precision_recall_curve(
        _scored_dict_test,
        anomaly_id_list=anomaly_ids
    )

    cur_auc = auc(
        recall,
        precison
    )
    print('AUC :: ', cur_auc)
    print('--------------------------')

    # return cur_auc, recall, precison

def process_2(
        CONFIG,
        _DIR,
        data_x,
        normal_ids,
        anomaly_ids
):
    model_obj = set_up_model(CONFIG, _DIR)

    _use_pretrained = CONFIG[_DIR]['use_pretrained']

    if _use_pretrained is True:

        saved_file_path = None
        pretrained_file = CONFIG[_DIR]['saved_model_file']

        print('Pretrained File :', pretrained_file)
        saved_file_path = os.path.join(
            SAVE_DIR,
            'checkpoints',
            pretrained_file
        )
        if saved_file_path is not None:
            model_obj.set_pretrained_model_file(saved_file_path)
        else:
            model_obj.train_model(data_x)

    elif _use_pretrained is False:
        model_obj.train_model(data_x)

    pairwise_cosine_dist = model_obj.get_pairwise_cosine_dist(data_x)
    print(pairwise_cosine_dist.shape)

    _list1 = list(range(1,2000))
    _list2 = list(range(25800,27800))

    _max = []
    _min = []
    _mean = []
    _var = []
    for _list in [_list1,_list2]:

        for i in _list :
            arr = pairwise_cosine_dist[i]
            _max.append(np.max(arr))
            _min.append(np.min(arr))
            _mean.append(np.mean(arr))
            _var.append(np.var(arr))

        import seaborn as sns
        import matplotlib.patches as mpatches
        cmaps = ['Reds', 'Blues', 'Greens', 'Greys']
        label_patches = []
        _res = [ _max,_min,_mean,_var ]
        _labels = ['_max', '_min', '_mean', '_variance']

        for i in range(4):
            offset = 3 * i
            label = _labels[i]
            ax = sns.kdeplot(
                _res[i],
                cbar = cmaps[i] + '_d'
            )
            label_patch = mpatches.Patch(
                color=sns.color_palette(cmaps[i])[2],
                label=label
            )
            label_patches.append(label_patch)
        # plt.legend(handles=label_patches, loc='upper left')
        ax.legend(_labels)
        plt.show()



def viz_tsne(data):
    from sklearn.manifold import TSNE

    X = np.array(data)
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=100,
        n_iter=500
    )

    tsne_results = tsne.fit_transform(X)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], c='g', s=18)
    plt.tight_layout()
    plt.show()


def main():
    global embedding_dims
    global SAVE_DIR
    global _DIR
    global DATA_DIR
    global CONFIG
    global CONFIG_FILE
    global MODEL_NAME
    global DOMAIN_DIMS

    time_1 = time.time()
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    _DIR = CONFIG['_DIR']
    DATA_DIR = CONFIG['DATA_DIR'] + '/' + _DIR
    setup_general_config()

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIR, 'checkpoints'))

    # ------------ #

    data_x, normal_ids, anomaly_ids = data_fetcher.get_data(
        DATA_DIR,
        _DIR
    )

    DOMAIN_DIMS = get_domain_dims()
    print('Data shape', data_x.shape)
    lof_1.KNN_K = CONFIG[_DIR]['lof_K']

    # ------------
    # 10 test cases
    # ------------

    time_1 = time.time()
    process(
        CONFIG,
        _DIR,
        data_x,
        normal_ids,
        anomaly_ids
    )

    time_2 = time.time()

    print('time taken ', time_2 - time_1)


# ----------------------------------------------------------------- #
# find out which model works best
# ----------------------------------------------------------------- #


main()

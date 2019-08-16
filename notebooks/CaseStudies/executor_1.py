import operator
import pickle
import numpy as np
import os
import sys
import time
import pprint
import inspect
from collections import OrderedDict
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import auc
import logging
import logging.handlers
import tensorflow as tf
import pandas as pd
tf.logging.set_verbosity(tf.logging.ERROR)

# matplotlib.use('Agg')
sys.path.append('./..')
sys.path.append('./../../.')



try:
    import src.m2_test_1layer.tf_model_3_withNorm as tf_model
except:
    from .src.m2_test_1layer import tf_model_3_withNorm as tf_model

try:
    from src.Eval import eval_v1 as eval
except:
    from .src.Eval import eval_v1 as eval



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
CONFIG_FILE = 'config_caseStudy_1.yaml'
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
logger = None


def setup_general_config():
    global MODEL_NAME
    global _DIR
    global SAVE_DIR
    global OP_DIR
    global _SAVE_DIR
    global CONFIG
    global logger
    SAVE_DIR = os.path.join(CONFIG['SAVE_DIR'], _DIR)
    OP_DIR = os.path.join(CONFIG['OP_DIR'], _DIR)
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
        embedding_dims = [config[_dir]['op_dims']]

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
        num_neg_samples=config[_dir]['num_neg_samples']
    )
    model_obj.set_l2_loss_flag(True)
    model_obj.inference = False
    model_obj.build_model()
    return model_obj

def get_data():
    global CONFIG
    global DATA_DIR
    global _DIR

    DIR = _DIR

    with open(os.path.join(
            CONFIG['DATA_DIR'],
            DIR,
            'domain_dims.pkl'
    ), 'rb') as fh:
        domain_dims = pickle.load(fh)

    train_x_pos_file = os.path.join(
        CONFIG['DATA_DIR'],
        DIR,
        'matrix_train_positive_v1.pkl'
    )

    with open(train_x_pos_file, 'rb') as fh:
        train_x_pos = pickle.load(fh)

    train_x_neg_file = os.path.join(
        CONFIG['DATA_DIR'],
        DIR,
        'negative_samples_v1.pkl'
    )

    with open(train_x_neg_file, 'rb') as fh:
        train_x_neg = pickle.load(fh)
        train_x_neg = train_x_neg

    test_x_file = os.path.join(
        CONFIG['DATA_DIR'],
        DIR,
        'matrix_test_positive.pkl'
    )

    with open(test_x_file, 'rb') as fh:
        test_x = pickle.load(fh)

    _df = pd.read_csv(os.path.join(CONFIG['DATA_DIR'],DIR,'test_data.csv'),header=0)
    test_id_list = list(_df['PanjivaRecordID'])

    return train_x_pos, train_x_neg, test_x, test_id_list, domain_dims


def process(
        CONFIG,
        _DIR,
        train_x_pos,
        train_x_neg,
        test_data_x,
        test_id_list
):
    global logger
    num_neg_samples = train_x_neg.shape[1]
    CONFIG[_DIR]['num_neg_samples'] = num_neg_samples
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
            model_obj.train_model(
                train_x_pos,
                train_x_neg
            )

    elif _use_pretrained is False:
        model_obj.train_model(
            train_x_pos,
            train_x_neg
        )

    print(' Len of test_ids ', len(test_id_list))
    print('Length of test data', test_data_x.shape)
    res = model_obj.get_event_score(test_data_x)
    print('Length of results ', len(res))


    res = list(res)
    _id_score_dict = {
        id: _res for id, _res in zip(
            test_id_list,
            res
        )
    }

    '''
    sort by ascending 
    since lower likelihood means anomalous
    '''
    tmp = sorted(
        _id_score_dict.items(),
        key=operator.itemgetter(1)
    )
    sorted_id_score_dict = OrderedDict()

    for e in tmp:
        sorted_id_score_dict[e[0]] = e[1][0]

    _ID = []
    _SCORE = []
    for k,v in sorted_id_score_dict.items():
        _ID.append(k)
        _SCORE.append(v)
    _df = pd.DataFrame(columns=['PanjivaRecordID','score'])
    _df['PanjivaRecordID'] = _ID
    _df['score'] = _SCORE
    _df.to_csv(os.path.join(OP_DIR,'result_1.csv'))
    # get embeddings
    emb_res = model_obj.get_record_embeddings(train_x_pos)
    with open(os.path.join(OP_DIR,'train_embeddings.pkl'),'wb') as fh:
        pickle.dump(emb_res,fh,pickle.HIGHEST_PROTOCOL)


    return


def main():
    global embedding_dims
    global SAVE_DIR
    global _DIR
    global DATA_DIR
    global CONFIG
    global CONFIG_FILE
    global MODEL_NAME
    global DOMAIN_DIMS
    global logger

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)


    DATA_DIR = os.path.join(CONFIG['DATA_DIR'], _DIR)

    setup_general_config()

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(
            os.path.join(SAVE_DIR, 'checkpoints')
        )

    # ------------ #

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIR, 'checkpoints'))

    # ------------ #
    logger.info('-------------------')


    train_x_pos, train_x_neg, test_x, test_id_list, domain_dims = get_data()
    process(
        CONFIG,
        _DIR,
        train_x_pos,
        train_x_neg,
        test_x,
        test_id_list
    )


    logger.info('-------------------')



# ----------------------------------------------------------------- #
# find out which model works best
# ----------------------------------------------------------------- #

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

try:
    log_file = 'case_studies_1.log'
except:
    log_file = 'm2.log'

_DIR = 'us_import'
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
OP_DIR = os.path.join(CONFIG['OP_DIR'], _DIR)

if not os.path.exists(CONFIG['OP_DIR']):
    os.mkdir(CONFIG['OP_DIR'])

if not os.path.exists(OP_DIR):
    os.mkdir(OP_DIR)

handler = logging.FileHandler(os.path.join(OP_DIR, log_file))
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info(' Info start ')
logger.info(' -----> ' + _DIR)
main()

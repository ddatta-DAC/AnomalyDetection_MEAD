
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
tf.logging.set_verbosity(tf.logging.ERROR)

# matplotlib.use('Agg')
sys.path.append('./..')
sys.path.append('./../../.')


try:
    from src.m2_test import tf_model_3 as tf_model
except:
    from .src.m2_test import tf_model_3 as tf_model

try:
    from src.Eval import eval_v1 as eval
except:
    from .src.Eval import eval_v1 as eval

try:
    from src.data_fetcher import data_fetcher as data_fetcher
except:
    from .src.data_fetcher import data_fetcher as data_fetcher



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

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    try:
        log_file = CONFIG['log_file']
    except:
        log_file = 'm1.log'
    handler = logging.FileHandler(os.path.join(OP_DIR,log_file))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(' Info start ')

    logger.info(pprint.pformat(CONFIG[_DIR], indent=4))
    return


# --------------------------------------------- #

def set_up_model(config, _dir, ablation_flag):
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

    loss_1_flag = True
    loss_2_flag = True
    loss_3_flag = True
    if ablation_flag:
        loss_1_flag = False
        loss_2_flag = False
        loss_3_flag = True

    model_obj.set_model_options(
        show_loss_figure=config[_dir]['show_loss_figure'],
        save_loss_figure=config[_dir]['save_loss_figure'],
        loss_1_flag=loss_1_flag,
        loss_2_flag=loss_2_flag,
        loss_3_flag=loss_3_flag
    )

    domain_dims = get_domain_dims()
    LR = config[_dir]['learning_rate']
    model_obj.set_model_hyperparams(
        domain_dims=domain_dims,
        emb_dims=embedding_dims,
        batch_size=config[_dir]['batchsize'],
        num_epochs=config[_dir]['num_epochs'],
        learning_rate=LR,
        num_neg_samples= config[_dir]['num_neg_samples']
    )
    model_obj.set_loss_lambda(1, 1, 1)
    model_obj.inference = False
    model_obj.build_model()
    return model_obj



def process(
        CONFIG,
        _DIR,
        train_x_pos,
        train_x_neg,
        test_pos,
        test_anomaly,
        ablation_flag
):
    global logger
    num_neg_samples = train_x_neg.shape[1]
    CONFIG[_DIR]['num_neg_samples'] = num_neg_samples
    model_obj = set_up_model(CONFIG, _DIR, ablation_flag)

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

    test_normal_ids = test_pos[0]
    test_anomaly_ids = test_anomaly[0]
    test_ids = list(np.hstack(
        [test_normal_ids,
         test_anomaly_ids]
    ))

    print(' Len of test_ids ', len(test_ids))
    test_normal_data = test_pos[1]
    test_anomaly_data = test_anomaly[1]
    test_data_x = np.vstack([
        test_normal_data,
        test_anomaly_data
    ])


    print('Length of test data', test_data_x.shape)
    res = model_obj.get_event_score(test_data_x)
    print('Length of results ', len(res))


    test_ids = list(test_ids)

    bounds = []
    training_pos_scores = model_obj.get_event_score(
        train_x_pos
    )
    training_pos_scores = [_[0] for _ in training_pos_scores]

    train_noise = np.reshape(train_x_neg, [-1, train_x_pos.shape[-1]])
    training_noise_scores = model_obj.get_event_score(
        train_noise
    )
    training_noise_scores = [_[0] for _ in training_noise_scores]

    bounds.append(min(training_noise_scores))
    bounds.append(max(training_pos_scores))

    print('Length of results ', len(res))

    res = list(res)
    _id_score_dict = {
        id: _res for id, _res in zip(test_ids, res)
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

    recall, precison = eval.precision_recall_curve(
        sorted_id_score_dict,
        anomaly_id_list=test_anomaly_ids,
        bounds=bounds
    )

    _auc = auc(recall, precison)
    logger.info('AUC')
    logger.info(str(_auc))

    print('--------------------------')




def main( exec_dir = None, ablation_flag = False ):
    global embedding_dims
    global SAVE_DIR
    global _DIR
    global DATA_DIR
    global CONFIG
    global CONFIG_FILE
    global MODEL_NAME
    global DOMAIN_DIMS
    global logger

    time_1 = time.time()
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if exec_dir is None:
        _DIR = CONFIG['_DIR']
    else:
        _DIR = exec_dir

    DATA_DIR = os.path.join(CONFIG['DATA_DIR'], _DIR)
    setup_general_config()

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIR, 'checkpoints'))

    # ------------ #
    logger.info('-------------------')
    logger.info( 'DIR ' + exec_dir)
    logger.info(' Ablation ')
    logger.info(ablation_flag)

    for _c in range(1,3+1) :
        logger.info(' >> c = ' + str(_c))
        train_x_pos, train_x_neg, test_pos, test_anomaly , domain_dims = data_fetcher.get_data_v3(
            CONFIG['DATA_DIR'],
            _DIR,
            c=_c
        )

        DOMAIN_DIMS = domain_dims
        print('Data shape', train_x_pos.shape)

        time_1 = time.time()
        process(
            CONFIG,
            _DIR,
            train_x_pos,
            train_x_neg,
            test_pos,
            test_anomaly,
            ablation_flag
        )

        time_2 = time.time()
    logger.info('-------------------')
    print('time taken ', time_2 - time_1)


# ----------------------------------------------------------------- #
# find out which model works best
# ----------------------------------------------------------------- #

for _exec_dir in ['china_import','peru_export','us_import'] :
    for _ablation_flag in [True, False] :
        main(_exec_dir , _ablation_flag )

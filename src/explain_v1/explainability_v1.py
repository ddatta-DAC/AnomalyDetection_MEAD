import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
import os
import logging
import logging.handlers
sys.path.append('./..')
sys.path.append('./../../.')

_DIR = 'china_export'
_FILE_DIR = './output'
DATA_DIR = './../../generated_data'

# ------------------------------ #

with open(os.path.join(_FILE_DIR, _DIR, 'train_embedding_values.pkl'), 'rb') as fh:
    arr = pickle.load(fh)

with open(os.path.join(DATA_DIR, _DIR, 'domain_dims.pkl'), 'rb') as fh:
    domain_dims = pickle.load(fh)

train_x = arr[0]
train_x_emb = arr[1]
emb_size = train_x_emb.shape[-1]

domain_sizes = list(domain_dims.values())
domain_names = list(domain_dims.keys())
domain_emb_dict = {}

domain_id2name = {e[0]: e[1] for e in enumerate(domain_names, 0)}
domain_name2id = {e[1]: e[0] for e in enumerate(domain_names, 0)}

for _ds, _dn in zip(domain_sizes, domain_names):
    a = np.zeros([_ds, emb_size])
    domain_emb_dict[domain_name2id[_dn]] = a


num_cols = train_x_emb.shape[1]
for _c in range(num_cols):
    ids = train_x[:, _c]
    vals = train_x_emb[:, _c, :]
    ids = np.reshape(ids, [-1, 1])
    tmp = np.hstack([ids, vals])
    _df = pd.DataFrame(tmp)
    _df = _df.rename(columns={0: 'id'})
    _df = _df.drop_duplicates(subset=['id'])

    for i, row in _df.iterrows():
        _id = int(row['id'])
        z = row.values
        domain_emb_dict[_c][_id] = z[1:]


def dissect_v4(arr):
    arr_sum = np.mean(arr, axis=0)
    res = []
    for i in range(arr.shape[0]):
        x = arr[i]
        prj = np.dot(arr_sum, x) / np.linalg.norm(arr_sum, ord=2)
        # print(prj)
        res.append(prj)
    res = np.array(res)
    return res


def get_embedding(id_value, domain_id, domain_emb_dict):
    return domain_emb_dict[domain_id][id_value]


def get_emb_arr(arr, domain_emb_dict):
    tmp = []
    for i in range(len(arr)):
        v = get_embedding(arr[i], i, domain_emb_dict)
        tmp.append(v)
    tmp = np.array(tmp)
    return tmp


def process(c, _random=False):
    global domain_emb_dict

    test_x_df_file = os.path.join(DATA_DIR, _DIR, 'test_data.csv')
    anomalies_c_df_file = os.path.join(DATA_DIR, _DIR, "anomalies_c{}_data.csv".format(c))
    anomalies_c_df = pd.read_csv(anomalies_c_df_file)

    def set_ref_id(row):
        _id = row['PanjivaRecordID']
        _id = str(_id)[:-3]
        return int(_id)

    anomalies_c_df['ref_id'] = 0
    anomalies_c_df['ref_id'] = anomalies_c_df.apply(
        set_ref_id,
        axis=1
    )

    test_df = pd.read_csv(test_x_df_file)
    anomalies_c_df = anomalies_c_df.loc[anomalies_c_df['ref_id'].isin(list(test_df['PanjivaRecordID']))]
    contrast_data = []

    for i, row in test_df.iterrows():
        t_row_copy = pd.Series(row, copy=True)
        _tmp = anomalies_c_df.loc[anomalies_c_df['ref_id'] == row['PanjivaRecordID']]
        del t_row_copy['PanjivaRecordID']
        v = t_row_copy.values
        del _tmp['PanjivaRecordID']
        del _tmp['ref_id']
        try:
            _tmp = (_tmp.values)[0]
            arr = np.vstack([v, _tmp])
            contrast_data.append(arr)
        except:
            pass

    l = 0
    i = 0
    tot_hr = 0

    for instance in contrast_data:
        i += 1
        n = instance[0]
        a = instance[1]
        diff_indices = np.nonzero(np.bitwise_xor(n, a))[0]
        # print('True   ', a)
        # print('Anomaly', n)
        n_emb = get_emb_arr(n, domain_emb_dict)
        a_emb = get_emb_arr(a, domain_emb_dict)

        prj_res = dissect_v4(a_emb)
        # print(prj_res)
        _mean = np.mean(prj_res)
        min_indices = np.where(prj_res < _mean)[0]
        l += len(min_indices)

        # select at random
        if _random:
            rnd = np.random.choice(np.array(range(0,a.shape[0])), int(a.shape[0]/2))
            min_indices = rnd

        # print(min_indices, diff_indices, set(min_indices).intersection(diff_indices))
        hr = len(set(min_indices).intersection(diff_indices)) / len(diff_indices)
        tot_hr += hr
        # norm_n = np.square(np.linalg.norm(np.sum(n_emb, axis=0), ord=2))
        # norm_a = np.square(np.linalg.norm(np.sum(a_emb, axis=0), ord=2))
        # print(np.tanh(norm_n), ' || ', np.tanh(norm_a))

    print(l / i)
    ahr = tot_hr / i
    print('Avg hit rate ', tot_hr / i)
    return  ahr

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
log_file = 'explainability_results_v2.log'

handler = logging.FileHandler(os.path.join(log_file))
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info(' Info start ')
logger.info(_DIR)


def main(RANDOM):
    global logger
    logger.info('-------------------')
    logger.info(RANDOM)
    res = []
    # ahr = process(1,RANDOM)
    # res.append(ahr)
    # logger.info('c ='+str(1))
    # logger.info(ahr)

    ahr = process(2,RANDOM)
    res.append(ahr)
    logger.info('c ='+str(2))
    logger.info(ahr)

    ahr = process(3,RANDOM)
    res.append(ahr)
    logger.info('c ='+str(3))
    logger.info(ahr)

    print(' Average ', np.mean(res))
    logger.info('Average over 2 ' + str(np.mean(res)))
    logger.info('-------------------')


main(RANDOM=False)
main(RANDOM=True)
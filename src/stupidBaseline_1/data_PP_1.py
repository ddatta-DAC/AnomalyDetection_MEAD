import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import glob
import pickle


if not os.path.exists('generated_data_b1'):
    os.mkdir('generated_data_b1')

def get_train_data(_DIR):
    op_path = os.path.join('generated_data_b1',_DIR)
    if not os.path.exists(op_path):
        os.mkdir(op_path)

    file = "./../../comprex/comprexData/{}/data_train.txt".format(_DIR)
    df = pd.read_csv(file, sep=' ',header=None)
    num_entities = np.max(df.values)+1
    op_file = os.path.join(op_path,'train_x.pkl')
    # check if data matrix is present
    if os.path.exists(op_file):
        with open(op_file,'rb') as fh:
            data_mat = pickle.load(fh)
        return data_mat


    data_mat = csr_matrix((len(df),num_entities),dtype=np.int16)

    for i,row in df.iterrows():
        vals = list(row.values)
        for v in vals :
            data_mat[i,v] = 1

    with open(op_file, 'wb') as fh:
        pickle.dump(data_mat, fh, pickle.HIGHEST_PROTOCOL)
    return data_mat

def get_test_data(_DIR,c=1):

    file = "./../../comprex/comprexData/{}/data_train.txt".format(_DIR)
    df = pd.read_csv(file, sep=' ',header=None)
    num_entities = np.max(df.values) + 1

    op_path = os.path.join('generated_data_b1', _DIR)
    op_test_x = os.path.join(op_path,'test_x_c{}.pkl'.format(c))

    file_data_test = "./../../comprex/comprexData/{}/test_data_c{}.txt".format(_DIR,c)
    file_all_id_test = "./../../comprex/comprexData/{}/id_test_set_c{}.txt".format(_DIR,c)
    file_anom_id_test ="./../../comprex/comprexData/{}/id_test_anomalies_c{}.txt".format(_DIR,c)

    if not os.path.exists(op_test_x):
        df = pd.read_csv(file_data_test, sep=' ',header=None)
        data_mat_test = csr_matrix((len(df), num_entities), dtype=np.int16)

        for i, row in df.iterrows():
            vals = list(row.values)
            for v in vals:
                data_mat_test[i, v] = 1
        with open(op_test_x,'wb') as fh:
            pickle.dump(data_mat_test,fh,pickle.HIGHEST_PROTOCOL)
    else:
        with open(op_test_x, 'rb') as fh:
            data_mat_test =  pickle.load(fh)

    df_id1 = pd.read_csv(file_all_id_test, sep=' ',header=None)
    all_ids = list(df_id1[0])

    df_id2 = pd.read_csv(file_anom_id_test, sep=' ',header=None)
    anom_ids = list(df_id2[0])

    return all_ids,anom_ids, data_mat_test


def get_data(_DIR,c=1):
    train_x = get_train_data(_DIR)
    all_ids,anom_ids, test_x = get_test_data(_DIR,c)
    return train_x,all_ids,anom_ids, test_x
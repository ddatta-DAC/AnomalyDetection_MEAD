import pickle
import os
import glob
import numpy as np

def get_data_v0(
        DATA_DIR,
        _DIR
):
    DATA_FILE = os.path.join(DATA_DIR, 'data.pkl')

    with open(DATA_FILE, 'rb') as fh:
        DATA_X = pickle.load(fh)

    x = DATA_X[0]
    normal_ids = DATA_X[1]
    anomaly_ids = DATA_X[2]
    return x, normal_ids, anomaly_ids


def get_data_v1(
        DATA_DIR,
        DIR
):
    with open(os.path.join(
            DATA_DIR,
            DIR,
            'domain_dims.pkl'
    ), 'rb') as fh:
        domain_dims = pickle.load(fh)


    train_x_pos_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_train_positive.pkl'
    )

    with open(train_x_pos_file, 'rb') as fh:
        train_x_pos = pickle.load(fh)

    train_x_neg_file = os.path.join(
        DATA_DIR,
        DIR,
        'ape_negative_samples.pkl'
    )

    with open(train_x_neg_file, 'rb') as fh:
        train_x_neg = pickle.load(fh)

    APE_term_2_file = os.path.join(
        DATA_DIR,
        DIR,
        'ape_term_2.pkl'
    )

    APE_term_4_file = os.path.join(
        DATA_DIR,
        DIR,
        'ape_term_4.pkl'
    )

    with open(APE_term_2_file, 'rb') as fh:
        APE_term_2 = pickle.load(fh)
        APE_term_2 = np.reshape(APE_term_2,[APE_term_2.shape[0],1])

    with open(APE_term_4_file, 'rb') as fh:
        APE_term_4 = pickle.load(fh)
        APE_term_4 = np.reshape(APE_term_4,[APE_term_4.shape[0],APE_term_4.shape[1],1])


    test_x_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_positive.pkl'
    )

    with open(test_x_file, 'rb') as fh:
        test_x = pickle.load(fh)

    anomaly_data_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_anomalies.pkl'
    )

    test_id_list_file = os.path.join(
        DATA_DIR,
        DIR,
        'test_idList.pkl'
    )

    with open(anomaly_data_file, 'rb') as fh:
        anomaly_data = pickle.load(fh)

    with open(test_id_list_file, 'rb') as fh:
        _id_list = pickle.load(fh)
        test_anomaly_idList = _id_list[0]
        test_normal_idList = _id_list[1]

    test_pos = [test_normal_idList, test_x]
    test_anomaly = [test_anomaly_idList, anomaly_data]
    return train_x_pos, train_x_neg, APE_term_2, APE_term_4, test_pos, test_anomaly , domain_dims




def get_data_v2(
        DATA_DIR,
        DIR
):
    with open(os.path.join(
            DATA_DIR,
            DIR,
            'domain_dims.pkl'
    ), 'rb') as fh:
        domain_dims = pickle.load(fh)


    train_x_pos_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_train_positive.pkl'
    )

    with open(train_x_pos_file, 'rb') as fh:
        train_x_pos = pickle.load(fh)

    train_x_neg_file = os.path.join(
        DATA_DIR,
        DIR,
        'ape_negative_samples.pkl'
    )

    with open(train_x_neg_file, 'rb') as fh:
        train_x_neg = pickle.load(fh)

    test_x_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_positive.pkl'
    )

    with open(test_x_file, 'rb') as fh:
        test_x = pickle.load(fh)

    anomaly_data_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_anomalies.pkl'
    )

    test_id_list_file = os.path.join(
        DATA_DIR,
        DIR,
        'test_idList.pkl'
    )

    with open(anomaly_data_file, 'rb') as fh:
        anomaly_data = pickle.load(fh)

    with open(test_id_list_file, 'rb') as fh:
        _id_list = pickle.load(fh)
        test_anomaly_idList = _id_list[0]
        test_normal_idList = _id_list[1]

    test_pos = [test_normal_idList, test_x]
    test_anomaly = [test_anomaly_idList, anomaly_data]
    return train_x_pos, train_x_neg, test_pos, test_anomaly , domain_dims



def get_data_v3(
        DATA_DIR,
        DIR
):
    with open(os.path.join(
            DATA_DIR,
            DIR,
            'domain_dims.pkl'
    ), 'rb') as fh:
        domain_dims = pickle.load(fh)


    train_x_pos_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_train_positive_v1.pkl'
    )

    with open(train_x_pos_file, 'rb') as fh:
        train_x_pos = pickle.load(fh)

    train_x_neg_file = os.path.join(
        DATA_DIR,
        DIR,
        'negative_samples_v1.pkl'
    )

    with open(train_x_neg_file, 'rb') as fh:
        train_x_neg = pickle.load(fh)

    test_x_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_positive.pkl'
    )

    with open(test_x_file, 'rb') as fh:
        test_x = pickle.load(fh)

    anomaly_data_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_anomalies.pkl'
    )

    test_id_list_file = os.path.join(
        DATA_DIR,
        DIR,
        'test_idList.pkl'
    )

    with open(anomaly_data_file, 'rb') as fh:
        anomaly_data = pickle.load(fh)

    with open(test_id_list_file, 'rb') as fh:
        _id_list = pickle.load(fh)
        test_anomaly_idList = _id_list[0]
        test_normal_idList = _id_list[1]

    test_pos = [test_normal_idList, test_x]
    test_anomaly = [test_anomaly_idList, anomaly_data]
    return train_x_pos, train_x_neg, test_pos, test_anomaly , domain_dims

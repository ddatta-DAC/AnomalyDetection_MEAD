import pandas as pd
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pickle
from sklearn.metrics import auc
from sklearn.decomposition import TruncatedSVD
import sys
import operator
from collections import OrderedDict
sys.path.append('./../.')
sys.path.append('/../../.')
# ---------- #
try:
    import data_PP_1 as data_fetcher
except:
    from src.stupidBaseline_1 import data_PP_1 as data_fetcher
try:
    from src.Eval import eval_v1 as eval
except:
    from .src.Eval import eval_v1 as eval


_DIR = 'peru_export'

train_x,all_ids, anom_ids, test_x  =  data_fetcher.get_data(_DIR,c=1)
tsvd = TruncatedSVD(n_components=100)
tsvd.fit(train_x)
_train_x = tsvd.transform(train_x)

# print('explained variance', tsvd.singular_values_)
clf = IsolationForest(
    max_samples=10000,
    contamination=0.0,
    n_estimators=1000,
    behaviour = 'old'
)
clf.fit(_train_x)


clf_1 = LocalOutlierFactor(
    n_neighbors=20,
    novelty =True
)
clf_1.fit(_train_x)

_test_x = tsvd.transform(test_x)
result = clf_1.score_samples(_test_x)

res = list(result)
_id_score_dict = {
    id: _res for id, _res in zip(all_ids, res)
}
tmp = sorted(
    _id_score_dict.items(),
    key=operator.itemgetter(1)
)
sorted_id_score_dict = OrderedDict()

for e in tmp:
    sorted_id_score_dict[e[0]] = e[1]

bounds = []
# training_pos_scores = clf.score_samples(
#     _train_x
# )
# training_pos_scores = [_[0] for _ in training_pos_scores]
from pprint import pprint

pprint(result)
bounds.append(-1)
bounds.append(0)
recall, precison = eval.precision_recall_curve(
    sorted_id_score_dict,
    anomaly_id_list=anom_ids,
    bounds=bounds
)

_auc = auc(recall, precison)
print('_auc',_auc)

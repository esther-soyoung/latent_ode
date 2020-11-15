# import sklearn as sk
import numpy as np
from sklearn.metrics import roc_auc_score

# truth
truth = [0]*2082 + [1]*318
truth = np.array(truth)

# plain
p_pred = [0]*1547 + [1]*535 + [0]*40 + [1]*278
p_pred = np.array(p_pred)

# reg
r_pred = [0]*1625 + [1]*457 + [0]*49 + [1]*269
r_pred = np.array(r_pred)

p_score = roc_auc_score(truth, p_pred)
r_score = roc_auc_score(truth, r_pred)
print(p_score)
print(r_score)

from load import *

fulld = full.todense()
n_train = 1842

# 50 cols with 1 <= x <= 100; uniform dist
col1 = np.array(np.where((np.min(fulld,axis=0) == 1) & (np.max(fulld,axis=0) == 100))[1]).flatten()
# 100 columns w/ unif dist 0 < x < 1
col2 = np.where(np.array(np.sum((fulld > 0) & (fulld < 1),axis=0)).flatten())[0]
# 7333 columns that are indicator variables (0 or 1)
col3 = np.array(np.where(np.sum((fulld != 0) & (fulld != 1),axis=0) == 0)[1]).flatten()
# the rest - 7129 cols power law dist
col4s = set(range(fulld.shape[1])).difference(set(col1).union(col2).union(col3))
col4 = np.array(sorted([x for x in col4s]))

# drop col1 and col2
full2 = fulld[:,np.hstack([col3,col4])]
del fulld
# find again indicator columns
ind_cols = np.array(np.where(np.sum((full2 != 0) & (full2 != 1),axis=0) == 0)[1]).flatten()
# find again count columns
count_cols = np.array(np.where(np.sum(full2 > 1,axis=0) > 0)[1]).flatten()

# only keep cols that have at least min_nnz non-zero values
#min_nnz = 150
#mask = (np.array(np.sum(full2[:n_train,:],axis=0)).flatten() >= min_nnz) & \
#       (np.array(np.sum(full2[n_train:,:],axis=0)).flatten() >= min_nnz)
min_nnz = 50
mask = np.array(np.sum(full2[n_train:,:],axis=0)).flatten() >= min_nnz
full3 = full2[:,mask]

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=.005)
complete("lr1",clf,full3[:n_train],y,full3[n_train:])

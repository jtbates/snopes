import os
import scipy.sparse
import numpy as np
import git
import datetime
from sklearn import cross_validation

repo = git.Repo(".",odbt=git.GitDB)
name_rev = repo.heads.master.commit.name_rev

FEATDIR = "data"
def read_sparse(fn,nrow,ncol):
    mat = scipy.sparse.lil_matrix((nrow,ncol))
    with open(os.path.join(FEATDIR,fn)) as f:
        for l in f:
            r, c, v = l.split()
            r, c, v = int(r)-1, int(c)-1, float(v)
            mat[(r,c)] = v
    return mat.tocsc()

training = read_sparse("training.txt",1842,26364)
testing = read_sparse("testing.txt",952,26364)
full = scipy.sparse.csc_matrix(np.vstack([training.todense(),testing.todense()]))

# get rid of features that are always zero
nnz_cols = np.where(np.array(full.sum(0)).flatten())[0]
training = training[:,nnz_cols]
testing = testing[:,nnz_cols]
full = full[:,nnz_cols]

label_training = np.zeros(1842)
with open(os.path.join(FEATDIR,"label_training.txt")) as f:
    for i,l in enumerate(f):
        label_training[i] = int(l)

X = training
y = label_training
     
def test(clf,X=X,y=y,k=5):
    scores = cross_validation.cross_val_score(clf, X, y, cv=k)
    print(scores)
    print(np.mean(scores))
    print(np.std(scores))

OUTDIR = "output"
INFODIR = "info"
def complete(name,clf,X=X,y=y,training=training):
    now = str(datetime.datetime.now())[5:16]
    fn = " ".join([name,now])
    scores = cross_validation.cross_val_score(clf, X, y, cv=10)
    info = ["commit: %s" % name_rev,
            "cv_scores: %s" % str(scores),
            "cv_score_mean: %s" % str(np.mean(scores)),
            "cv_score_std: %s" % str(np.std(scores))
           ]
    with open(os.path.join(INFODIR,fn),'w') as f:
        for l in info:
            print(l)
            f.write(l+"\n")

    clf.fit(X,y)
    z = clf.predict(training)
    with open(os.path.join(OUTDIR,fn),'w') as f:
        f.writelines([str(int(p)) + "\r\n" for p in z])

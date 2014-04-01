from load import *

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import TruncatedSVD

N = full.shape[0]
idf = np.log(N / np.diff(full.indptr))
idf_diag = scipy.sparse.lil_matrix((len(idf),len(idf)))
idf_diag.setdiag(idf)
tfidf = full * idf_diag

lsa_k = 100
lsa = TruncatedSVD(n_components=lsa_k)
lsa.fit(tfidf)
Xt = lsa.transform(X*idf_diag)
clf = svm.SVC(kernel='linear', C=0.01)

testingt = lsa.transform(testing*idf_diag)
complete("lsa_svm1",clf,Xt,y,testingt)

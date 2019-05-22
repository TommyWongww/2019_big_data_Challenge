# @Time    : 2019/5/21 16:42
# @Author  : shakespere
# @FileName: baseline.py
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm,metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
column = "review"
train = pd.read_csv('./data/train.csv',lineterminator='\n')
test = pd.read_csv('./data/20190520_test.csv',lineterminator='\n')
test_id = test["ID"].copy()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
train_data = trn_term_doc
test_data = test_term_doc
fid0=open('./data/result0.csv','w')

label = train["label"]
train["predict"] = [0 if item=='Negative' else 1 for item in label]
train_label=(train["predict"]).astype(int)

folds = StratifiedKFold(n_splits=10,shuffle=False,random_state=2019)
predictions = np.zeros(test_id.shape[0])
aucs = []
for fold_, (train_index,test_index) in enumerate(folds.split(train_data,train_label)):
    print("Fold:{}".format(fold_ + 1))
    cv_train_data,cv_train_label = train_data[train_index],train_label[train_index]
    cv_test_data,cv_test_label = train_data[test_index],train_label[test_index]

    lin_clf = svm.LinearSVC()
    lin_clf = CalibratedClassifierCV(lin_clf,cv=5)
    lin_clf.fit(cv_train_data,cv_train_label)
    test_predict = lin_clf.predict_proba(cv_test_data)[:, 1]
    auc = metrics.roc_auc_score(cv_test_label,test_predict)
    predictions += lin_clf.predict_proba(test_data)[:,1] / folds.n_splits
    aucs.append(auc)
    print("auc score: %.5f" % auc)
print("Mean auc",np.mean(aucs))

i=1
fid0.write("ID,Pred"+"\n")
for item in predictions:
    fid0.write(str(i)+","+str(item)+"\n")
    i=i+1
fid0.close()
# clf = svm.SVC(C=5.0)
# clf.fit(trn_term_doc,y)
# predict_prob_y = clf.predict_proba(test_term_doc)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
# #end svm ,start metrics
# test_auc = metrics.roc_auc_score(test_y,predict_prob_y)#验证集上的auc值
# print(test_auc)

# @Time    : 2019/5/21 19:18
# @Author  : shakespere
# @FileName: baseline1.py
import csv
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import svm,metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings;warnings.filterwarnings('ignore')

# read data
train = pd.read_csv("./data/train.csv", lineterminator='\n', header=0)
# print(train.head(5))
train['label'] = train['label'].map({'Negative':0, 'Positive': 1})
# print(train.head(5))
# print(train.isnull().sum())

test = pd.read_csv("./data/20190520_test.csv", lineterminator='\n', header=0)
# print(test.isnull().sum())

words = []
for _ in train['review'].values:
    words.append(' '.join(WordPunctTokenizer().tokenize(_)))
train_data = words
# print(train_data[0:5])
train_label = np.array(train['label'].values, dtype='int8')

words = []
for _ in test['review'].values:
    words.append(' '.join(WordPunctTokenizer().tokenize(_)))
test_data = words

ngram = 2
vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram), max_df=0.9)
corpus_all = train_data + test_data
vectorizer.fit(corpus_all)
corpus_all = vectorizer.transform(corpus_all)


lentrain = len(train_data)
train_data = corpus_all[:lentrain]
test_data = corpus_all[lentrain:]

# model training and test

folds = StratifiedKFold(n_splits=30, shuffle=False, random_state=2019)
predictions = np.zeros(test_data.shape[0])

aucs = []
for fold_, (train_index, test_index) in enumerate(folds.split(train_data, train_label)):
    print("Fold :{}".format(fold_ + 1))
    cv_train_data, cv_train_label= train_data[train_index], train_label[train_index]
    cv_test_data, cv_test_label = train_data[test_index], train_label[test_index]

    # Logistic Regression
    # model = LR(solver='lbfgs')
    # model.fit(cv_train_data, cv_train_label)
    # auc = metrics.roc_auc_score(cv_test_label, model.predict_proba(cv_test_data)[:, 1])
    # predictions += model.predict_proba(test_data)[:, 1] / folds.n_splits

    # SGD classifier
    # model = LogisticRegression(solver="lbfgs",max_iter=3000)
    model = SGD(alpha=0.00001, penalty='l2', tol=10000, shuffle=True, loss='log')
    # 朴素贝叶斯
    # model = MultinomialNB()
    #k近邻
    # model = KNeighborsClassifier()
    # model = svm.LinearSVC()
    # 随机森林
    # model = RandomForestClassifier()

    # model = CalibratedClassifierCV(model, cv=5)
    model.fit(cv_train_data, cv_train_label)
    auc = metrics.roc_auc_score(cv_test_label, model.predict_proba(cv_test_data)[:, 1])
    predictions += model.predict_proba(test_data)[:, 1] / folds.n_splits

    # model = SVC(gamma='auto', probability=True)
    # model.fit(cv_train_data, cv_train_label)
    # auc = metrics.roc_auc_score(cv_test_label, model.predict_proba(cv_test_data)[:, 1])
    # predictions += model.predict_proba(test_data)[:, 1] / folds.n_splits

    aucs.append(auc)
    print("auc score: %.5f" % auc)

print('Mean auc', np.mean(aucs))
predictions = pd.DataFrame(predictions)
id = pd.DataFrame(np.arange(1, len(predictions) + 1))
data = pd.concat([id, predictions], axis=1)
data.to_csv('./data/merge_{}_predictions.csv'.format(np.mean(aucs)), header=['ID', 'Pred'], index=False)
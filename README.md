# 2019_big_data_Challenge
## 预选赛
# baseline
- 是2019.05.21晚上写的最初的版本，这个版本在0520-0526的数据集上public得分：0.86271889，但是无法通过private
# baseline1 
- 是2019.05.22上午写的另一个版本，里面有Logistic Regression、SGD classifier、朴素贝叶斯、k近邻、LinearSVC、RandomForestClassifier
# baseline3 
- 是另写的深度模型版本，这系列没有特别优化，然后因为预选赛 数据量也不是很大，效果非常一般，可以参考我kaggle上的toxic系列赛的[kernel->](https://www.kaggle.com/shakespere/keras-baseline-lstm-att-5-fold-bn-dp-2embedding-l),模型中的embedding文件也可以从上面获取
# result
- 实验效果是baseline1效果最好，其中的朴素贝叶斯效果最佳，最后通过成绩是用朴素贝叶斯和SGD，分别10折、15折、20折、30折 共产生8组predictions提交文件，然后用sum文件进行合并，这里只是简单平均融合，线上0.87080826，通过private


## 赛题介绍
预选赛 官网介绍->[https://www.kesci.com/home/competition/5cb80fd312c371002b12355f/content/1]
## 预选赛题——文本情感分类模型
- 本预选赛要求选手建立文本情感分类模型，选手用训练好的模型对测试集中的文本情感进行预测，判断其情感为「Negative」或者「Positive」。所提交的结果按照指定的评价指标使用在线评测数据进行评测，达到或超过规定的分数线即通过预选赛。

## ToDo：
这个比赛还是比较有意思的，通过预选赛后，后续有时间会跟进 正式赛题的 baseline。

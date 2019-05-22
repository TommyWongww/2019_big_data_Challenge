# 2019_big_data_Challenge
## 预选赛
# baseline
- 是2019.05.21晚上写的最初的版本，这个版本在0520-0526的数据集上public得分：0.86271889，但是无法通过private
# baseline1 
- 是2019.05.22上午写的另一个版本，里面有Logistic Regression、SGD classifier、朴素贝叶斯、k近邻、LinearSVC、RandomForestClassifier
# baseline3 
- 是另写的深度模型版本，这系列没有特别优化，然后因为预选赛 数据量也不是很大，效果非常一般
# result
- 实验效果是baseline1效果最好，其中的朴素贝叶斯效果最佳，最后通过成绩是用朴素贝叶斯和SGD，分别10折、15折、20折、30折 共产生8组predictions提交文件，然后用sum文件进行合并，这里只是简单平均融合，线上0.87080826，通过private

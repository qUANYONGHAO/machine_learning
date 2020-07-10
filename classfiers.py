
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:30:37 2019

@author: Litengfei
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
#设置中文字体
plt.rcParams['font.sans-serif']=['SimHei']

def load_data(name):

    dataset = name()
    X = dataset.data
    Y = dataset.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)


#a = []
#b = ['决策树','随机森林','KNN','贝叶斯','SVM','Adaboost','Logistic','GBDT']
#X = wine.data
#y = wine.target

#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
#forest = RandomForestClassifier(n_estimators=6,random_state=3)
#forest.fit(X_train,y_train)

#决策树
def dtc_classfier():
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    dtc = DecisionTreeClassifier(#criterion="entropy"
#                                 ,random_state=30
#                                 ,splitter="random"
#                                 ,max_depth=3
                                 )
    parameters = {'criterion':('entropy', 'gini'), 'random_state':[10, 20, 30,40], 'max_depth':[3,4,5,6]}
    dtc = GridSearchCV(dtc, parameters)
    dtc = dtc.fit(X_train,Y_train)
    print(dtc.best_params_)
    score_d = dtc.score(X_test,Y_test)
    print('得分为:',score_d)
    pre_d = dtc.predict(X_test)
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_d)#决策树
    plt.title('决策树')
    plt.show()
    print(confusion_matrix(Y_test,pre_d))
    print(classification_report(Y_test,pre_d))
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('决策树',Y_test,pre_d)


#wine_types = np.unique(Y)
#n_class = wine_types.size
#y_one_hot = label_binarize(Y_test, np.arange(n_class))
#fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),pre_d.ravel())
#auc = metrics.auc(fpr, tpr)
#mpl.rcParams['font.sans-serif'] = u'SimHei'
#mpl.rcParams['axes.unicode_minus'] = False
#            #FPR就是横坐标,TPR就是纵坐标
#plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
#plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
#plt.xlim((-0.01, 1.02))
#plt.ylim((-0.01, 1.02))
#plt.xticks(np.arange(0, 1.1, 0.1))
#plt.yticks(np.arange(0, 1.1, 0.1))
#plt.xlabel('False Positive Rate', fontsize=13)
#plt.ylabel('True Positive Rate', fontsize=13)
#plt.grid(b=True, ls=':')
#plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
#plt.show()


    #a.append(score_d)
#随即森林
def rfc_classfier():
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    rfc = RandomForestClassifier()
    parameters = {'n_estimators':[5,10,15,20,25,30], 'random_state':[10, 20, 30,40,50,60]}
    rfc = GridSearchCV(rfc, parameters)
    #print(rfc.best_params_)
    rfc = rfc.fit(X_train,Y_train)
    score_r = rfc.score(X_test,Y_test)
    print('得分为:',score_r)
    pre_r = rfc.predict(X_test)
    print(confusion_matrix(Y_test,pre_r))
    print(classification_report(Y_test,pre_r))
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_r)#随机森林
    plt.title('随机森林')
    plt.show()
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('随机森林',Y_test,pre_r)
    '''  
    f1 = f1_score(Y_test, pre_r)
    print('模型精确度:',p)
    print('模型召回率:',r)
    print('F1_score:',f1)
    '''
    #a.append(score_r)
#KNN分类
def knn_classfier():
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    knn = KNeighborsClassifier(#algorithm='auto'
                               #,n_neighbors=10
                               #,weights='uniform'
                               )
    parameters = {'algorithm':('auto','ball_tree','kd_tree','brute'),'n_neighbors':[3,4,5,6,7,8,9,10],'weights':('uniform','distance')}
    knn = GridSearchCV(knn, parameters)
    knn = knn.fit(X_train, Y_train)
    score_k = knn.score(X_test,Y_test)
    print('得分为:',score_k)
    pre_k = knn.predict(X_test)
    print(confusion_matrix(Y_test,pre_k))
    print(classification_report(Y_test,pre_k))
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_k)#KNN分类
    plt.title('KNN')
    plt.show()
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('KNN',Y_test,pre_k)
    '''
    p = precision_score(Y_test, pre_k)
    r = recall_score(Y_test, pre_k)  
    f1 = f1_score(Y_test, pre_k)
    print('模型精确度:',p)
    print('模型召回率:',r)
    print('F1_score:',f1)
    '''
    #a.append(score_k)
#高斯贝叶斯
def gnb_classfier():
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    gnb = GaussianNB().fit(X_train,Y_train)
    score_G = gnb.score(X_test,Y_test)
    print('得分为:',score_G)
    pre_G = gnb.predict(X_test)
    print(confusion_matrix(Y_test,pre_G))
    print(classification_report(Y_test,pre_G))
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_G)#高斯贝叶斯
    plt.title('贝叶斯')
    plt.show()
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('贝叶斯',Y_test,pre_G)
    '''
    p = precision_score(Y_test, pre_G)
    r = recall_score(Y_test, pre_G)  
    f1 = f1_score(Y_test, pre_G)
    print('模型精确度:',p)
    print('模型召回率:',r)
    print('F1_score:',f1)
    '''
    #a.append(score_G)
#SVM分类器
def svm_classfier():
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    svm = SVC(#C=0.8
              #,kernel='rbf'
              #,gamma=0.1
              )
    parameters = [
    {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf']
    },
    {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'kernel': ['linear']
    }
]
    svm = GridSearchCV(svm, parameters)
    svm = svm.fit(X_train,Y_train)
    score_s = svm.score(X_test,Y_test)
    print('得分为:',score_s)
    pre_s = svm.predict(X_test)
    print(confusion_matrix(Y_test,pre_s))
    print(classification_report(Y_test,pre_s))
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_s)#SVM分类
    plt.title('SVM')
    plt.show()
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('SVM',Y_test,pre_s)
#    p = precision_score(Y_test, pre_s)
#    r = recall_score(Y_test, pre_s)
#    f1 = f1_score(Y_test, pre_s)
#    print('模型精确度:',p)
#    print('模型召回率:',r)
#    print('F1_score:',f1)
    #a.append(score_s)
#Adaboost
def adb_classfier():
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    adb = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=4
#            ,min_samples_split=20
#            ,min_samples_leaf=5
#            ,algorithm="SAMME"
#            ,n_estimators=200
#            ,learning_rate=0.8
            ))
    parameters = {'n_estimators':[30,50,70,90,120,150,200],'learning_rate':[0.8,1,1.2,1.4,1.6]}
    adb = GridSearchCV(adb, parameters)
    adb = adb.fit(X_train, Y_train)
    score_a = adb.score(X_test,Y_test)
    print('得分为:',score_a)
    pre_a = adb.predict(X_test)
    print(confusion_matrix(Y_test,pre_a))
    print(classification_report(Y_test,pre_a))
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_a)#Adaboost
    plt.title('Adaboost')
    plt.show()
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('Adaboost',Y_test,pre_a)
#    p = precision_score(Y_test, pre_a)
#    r = recall_score(Y_test, pre_a)
#    f1 = f1_score(Y_test, pre_a)
#    print('模型精确度:',p)
#    print('模型召回率:',r)
#    print('F1_score:',f1)
    #a.append(score_a)
#逻辑回归
def lr_classfier():
    '''
    lr = LogisticRegressionCV(multi_class="ovr"
                              ,fit_intercept=True
                              ,Cs=np.logspace(-2,2,20)
                              ,cv=2,penalty="l2"
                              ,solver="lbfgs"
                              ,tol=0.01)
    '''
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    lr = LR(#penalty="l1"
             solver="liblinear"
#            ,C=0.8
#            ,max_iter=1000
            )
    parameters = {'penalty':('l1','l2'),'C':[0.4,0.6,0.8,1],'max_iter':[100,200,500,800,1000]}
    lr = GridSearchCV(lr, parameters)
    lr = lr.fit(X_train,Y_train)
    score_l = lr.score(X_test,Y_test)
    print('得分为:',score_l)
    pre_l = lr.predict(X_test)
    print(confusion_matrix(Y_test,pre_l))
    print(classification_report(Y_test,pre_l))
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_l)#逻辑回归i
    plt.title('Logistic')
    plt.show()
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('逻辑回归',Y_test,pre_l)
#    p = precision_score(Y_test, pre_l)
#    r = recall_score(Y_test, pre_l)
#    f1 = f1_score(Y_test, pre_l)
#    print('模型精确度:',p)
#    print('模型召回率:',r)
#    print('F1_score:',f1)
    #a.append(score_l)
#GBDT分类
def gdbt_classfier():
    shu_jv_ji_1=['1.鸢尾花','2.白酒','3.乳腺癌']
    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = load_iris()
    elif xvanze == 2:
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    gbdt = GradientBoostingClassifier(random_state=30)
    parameters = {'n_estimators':range(20,201,20),'max_depth':range(3,14,2),'learning_rate':[0.1,0.2,0.4,0.6,0.8]}
    gbdt = GridSearchCV(gbdt, parameters)
    gbdt = gbdt.fit(X_train,Y_train)
    score_g = gbdt.score(X_test,Y_test)
    print('得分为:',score_g)
    pre_g = gbdt.predict(X_test)
    print(confusion_matrix(Y_test,pre_g))
    print(classification_report(Y_test,pre_g))
    plt.scatter(X_test[:,0],X_test[:,1],c=pre_g)#GDBT梯度
    plt.title('GDBT')
    plt.show()
    b = input('请选择是否画出ROC曲线！(y/n):')
    if b == 'y':
        plot_roc('GDBT',Y_test,pre_g)
#    p = precision_score(Y_test, pre_g)
#    r = recall_score(Y_test, pre_g)
#    f1 = f1_score(Y_test, pre_g)
#    print('模型精确度:',p)
#    print('模型召回率:',r)
#    print('F1_score:',f1)
    #a.append(score_g)

#ROC曲线
def plot_roc(title,labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

def main():
    while(1):

        sufa_xvuanze=['1.决策树','2.随机森林','3.KNN','4.贝叶斯','5.SVM','6.Adaboost','7.Logistic','8.GDBT']

        print(sufa_xvuanze)
        sufa_xvuanze=int(input('请选择要是用的算法(用前面数字代替即可):'))
        if sufa_xvuanze == 1:
            dtc_classfier()
        elif sufa_xvuanze == 2:
            rfc_classfier()
        elif sufa_xvuanze == 3:
            knn_classfier()
        elif sufa_xvuanze == 4:
            gnb_classfier()
        elif sufa_xvuanze == 5:
            svm_classfier()
        elif sufa_xvuanze == 6:
            adb_classfier()
        elif sufa_xvuanze == 7:
            lr_classfier()
        elif sufa_xvuanze == 8:
            gdbt_classfier()

        a = input('请选择是否重新开始！(y/n):')
        if a == 'y':
            continue
        else:
            break
    print('感谢使用！')
if __name__ == '__main__':
    main()
'''
#画出分类结果图
plt.figure(figsize=(25,8))
ax1 = plt.subplot(241)
ax2 = plt.subplot(242)
ax3 = plt.subplot(243)
ax4 = plt.subplot(244)
ax5 = plt.subplot(245)
ax6 = plt.subplot(246)
ax7 = plt.subplot(247)
ax8 = plt.subplot(248)

plt.sca(ax1)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_d)#决策树
#plt.show()
plt.sca(ax2)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_r)#随机森林
#plt.show()
plt.sca(ax3)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_k)#KNN分类
#plt.show()
plt.sca(ax4)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_G)#高斯贝叶斯
#plt.show()
plt.sca(ax5)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_s)#SVM分类
#plt.show()
plt.sca(ax6)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_a)#Adaboost
#plt.show()
plt.sca(ax7)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_l)#逻辑回归i
#plt.show()
plt.sca(ax8)
plt.scatter(X_test[:,0],X_test[:,1],c=pre_g)#GDBT梯度

plt.show()


plt.figure(figsize=(20,8),dpi=80)
x = [1,2,3,4,5,6,7,8]

group_labels = ['决策树','随机森林','KNN','贝叶斯','SVM','Adaboost','Logistic','GBDT']
plt.plot(x,a)
plt.xticks(x, group_labels, rotation=0)
plt.grid()
plt.show()
'''


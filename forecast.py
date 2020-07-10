
# coding: utf-8
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import matplotlib.font_manager as fm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import xgboost as xgb
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch  #爬山算法
from pgmpy.estimators import BicScore
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

def jve_ce_shu():
    warnings.filterwarnings("ignore")

    print('现在进行的算法是决策树回归')
    #数据集选择

    shu_jv_ji_1=['1.housing1.CSV','2.forestfires.csv']

    print(shu_jv_ji_1)
    xvanze=int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = pd.read_csv("housing1.CSV")
    else:
        data = pd.read_csv("forestfires.csv")

    #决策树最大深度选择
    depth = int(input("请输入决策树最大深度(如：'4')："))
    decision_regressor = DecisionTreeRegressor(max_depth=depth)

    #决策树回归
    X = data.iloc[:,-2]
    Y = data.iloc[:,-1]
    X = np.array(X).reshape(np.shape(X)[0], 1)
    Y = np.array(Y).reshape(np.shape(Y)[0], 1)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=30)
    # 训练数据
    decision_regressor.fit(train_X, train_Y)

    # 使用测试集来判断好坏
    predict_test_Y = decision_regressor.predict(test_X)

    # 决策树模型得分指标   （以测试集为标准进行估测）

    print('均方误差:{}'.format(metrics.mean_squared_error(predict_test_Y, test_Y)))
    print('平均结对误差:{}'.format(metrics.mean_absolute_error(predict_test_Y, test_Y)))
    print('解释方差分:{}'.format(metrics.explained_variance_score(predict_test_Y, test_Y)))
    print('R2得分:{}'.format(metrics.r2_score(predict_test_Y, test_Y)))

    # 画图
    myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\STKAITI.ttf')
    X_test = np.arange(np.min(X), np.max(X), 0.01)[:, np.newaxis]
    Y_1 = decision_regressor.predict(X_test)
    plt.figure()
    plt.scatter(X[0:10], Y[0:10], s=20, edgecolor="black", c="darkorange", label="数据")
    plt.plot(X_test, Y_1, color="cornflowerblue", label="决策树回归", linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("决策树回归", fontproperties=myfont, fontsize=20)
    plt.legend(prop=myfont)
    plt.show()

    y_train_pred = decision_regressor.predict(train_X)
    y_test_pred = decision_regressor.predict(test_X)
    y_train_pred = np.array(y_train_pred).reshape(np.shape(y_train_pred)[0], 1)
    y_test_pred = np.array(y_test_pred).reshape(np.shape(y_test_pred)[0], 1)
    # 残差评估方法
    plt.scatter(y_train_pred, y_train_pred - train_Y,
                c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - test_Y,
                c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=np.min(y_train_pred), xmax=np.max(y_train_pred), lw=2, colors='red')
    plt.xlim([-10, 50])
    plt.show()
    # 折线图
    plt.figure()
    X_linear = np.arange(0, len(y_test_pred), 1)[:, np.newaxis]
    plt.plot(X_linear, test_Y, color="red", label="原始数据", lw=2, linestyle="-")
    plt.plot(X_linear, y_test_pred, color="green", label="预测数据", lw=2, linestyle="-")
    plt.show()

def duo_xiang_shi():
    warnings.filterwarnings("ignore")
    print('现在进行的算法是多项式回归')
    #数据集选择

    shu_jv_ji_2 = ['1.housing1.CSV', '2.forestfires.csv']

    print(shu_jv_ji_2)
    xvanze = int(input('请选择要使用数据集(用前面数字代替即可):'))
    if xvanze == 1:
        data = pd.read_csv("housing1.CSV")
    else:
        data = pd.read_csv("forestfires.csv")

    #多项式回归
    X = data.iloc[:, :-1].values
    y = data[['medv']].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    # 线性模型训练
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    # 多项式回归模型构造及训练
    spr = LinearRegression()
    quadratic = PolynomialFeatures()
    X_train_quad = quadratic.fit_transform(X_train)
    spr.fit(X_train_quad, y_train)
    y_train_pred_quad = spr.predict(X_train_quad)
    y_test_pred_quad = spr.predict(quadratic.fit_transform(X_test))

    # 残差评估方法
    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, colors='red')
    plt.xlim([-10, 50])
    plt.show()
    # 折线图
    plt.figure()
    X_linear = np.arange(0, len(y_test_pred), 1)[:, np.newaxis]
    plt.plot(X_linear, y_test, color="red", label="原始数据", lw=2, linestyle="-")
    plt.plot(X_linear, y_test_pred, color="green", label="预测数据", lw=2, linestyle="-")
    plt.show()

    # 均方误差评价指标
    from sklearn.metrics import mean_squared_error

    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred))
          )
    print('PolynomialFeatures MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred_quad),
        mean_squared_error(y_test, y_test_pred_quad))
          )

    #决定系数评价指标
    from sklearn.metrics import r2_score

    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),
           r2_score(y_test, y_test_pred)))
    print('PolynomialFeatures R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred_quad),
           r2_score(y_test, y_test_pred_quad)))

    print('均方误差:{}'.format(metrics.mean_squared_error(y_test_pred_quad, y_test)))
    print('平均结对误差:{}'.format(metrics.mean_absolute_error(y_test_pred_quad, y_test)))
    print('解释方差分:{}'.format(metrics.explained_variance_score(y_test_pred_quad, y_test)))
    print('R2得分:{}'.format(metrics.r2_score(y_test_pred_quad, y_test)))

def xg_yv_ce():
    warnings.filterwarnings("ignore")
    print('现在进行的算法是XGboost')
    boston = load_boston()
    X, y = boston.data, boston.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=14)
    print("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)
    # 1. 参数构建
    params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
    num_round = 2
    # 2. 模型训练
    bst = xgb.train(params, dtrain, num_round)
    # 3. 模型保存
    bst.save_model('xgb.model')
    y_pred = bst.predict(dtest)

    # 4. 加载模型
    bst2 = xgb.Booster()
    bst2.load_model('xgb.model')
    # 5 使用加载模型预测
    y_pred2 = bst2.predict(dtest)
    print('均方误差:{}'.format(metrics.mean_squared_error(y_pred2, y_test)))
    print('平均结对误差:{}'.format(metrics.mean_absolute_error(y_pred2, y_test)))
    print('解释方差分:{}'.format(metrics.explained_variance_score(y_pred2, y_test)))
    print('R2得分:{}'.format(metrics.r2_score(y_pred2, y_test)))

    plt.figure(figsize=(12, 6), facecolor='w')
    ln_x_test = range(len(x_test))

    myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\STKAITI.ttf')
    plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'实际值')
    plt.plot(ln_x_test, y_pred, 'g-', lw=4, label=u'XGBoost模型')
    plt.xlabel(u'数据编码', fontproperties=myfont)
    plt.ylabel(u'租赁价格', fontproperties=myfont)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.title(u'波士顿房屋租赁数据预测', fontproperties=myfont)
    plt.show()

    # 找出最重要的特征
    plot_importance(bst, importance_type='cover')
    plt.show()

def bei_ye_si():
    warnings.filterwarnings("ignore")
    print('现在进行的算法是贝叶斯网络')
    f = open('泰坦尼克号.txt')
    dataset = pd.read_table(f, delim_whitespace=True)
    train = dataset[:800]
    test = dataset[800:]
    hc = HillClimbSearch(train, scoring_method=BicScore(train))
    best_model = hc.estimate()
    best_model.fit(train, estimator=BayesianEstimator, prior_type="BDeu")  # default equivalent_sample_size=5
    predict_data = test.drop(columns=['Survived'], axis=1)
    y_pred = best_model.predict(predict_data)
    print((y_pred['Survived'] == test['Survived']).sum() / len(test))  # 测试集精度'''

def ma_er_ke_fu():
    warnings.filterwarnings("ignore")
    print('现在进行的算法是马尔科夫')
    # 加载数据集
    data_path = 'D:\PycharmProjects\predict_system\马尔科夫\data_hmm.txt'
    df = pd.read_csv(data_path, header=None)
    print(df.info())  # 查看数据信息，确保没有错误
    print(df.head())
    print(df.tail())

    # 画出原始数据的走势图
    df.iloc[:, 2].plot()

    dataset_X = df.iloc[:, 2].values.reshape(1, -1).T  # 前面两列是日期，用第2列作数据集
    # 需要构建成二维数组形式，故而需要加上一个轴
    print(dataset_X.shape)  # 有3312个训练样本组成一列

    # 建立HMM模型，并训练

    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
    model.fit(dataset_X)

    # 预测其状态
    hidden_states = model.predict(dataset_X)

    for i in range(model.n_components):  # 打印出每个隐含状态
        mean = model.means_[i][0]
        variance = np.diag(model.covars_[i])[0]
        print('Hidden state: {}, Mean={:.3f}, Variance={:.3f}'
              .format((i + 1), mean, variance))

    # 使用HMM模型生成数据
    N = 1000
    samples, _ = model.sample(N)
    plt.plot(samples[:, 0])

    # 模型的提升，修改n_components

    for i in [8, 12, 16, 18, 20]:
        model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000)
        model.fit(dataset_X)
        samples, _ = model.sample(1000)
        plt.plot(samples[:, 0])
        plt.title('hidden state N={}'.format(i))
        plt.show()

def main():
    warnings.filterwarnings("ignore")
    while(1):
        print('请选择你要运行的算法！')
        sufa_xvuanze=['1.决策树回归','2.多项式回归','3.XGboost','4.贝叶斯网络','5.马尔科夫']

        print(sufa_xvuanze)
        sufa_xvuanze=int(input('请选择要是用的算法(用前面数字代替即可):'))
        if sufa_xvuanze == 1:
            jve_ce_shu()
        elif sufa_xvuanze == 2:
            duo_xiang_shi()
        elif sufa_xvuanze == 3:
            xg_yv_ce()
        elif sufa_xvuanze == 4:
            bei_ye_si()
        elif sufa_xvuanze == 5:
            ma_er_ke_fu()

        a = input('请选择是否重新开始！(y/n):')
        if a == 'y':
            continue
        else:
            break
    print('感谢使用！')
if __name__ == '__main__':
    main()

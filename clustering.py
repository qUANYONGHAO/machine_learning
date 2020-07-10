import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import seaborn as sns; sns.set()
from matplotlib.patches import Ellipse
from sklearn import mixture
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_completeness_v_measure
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import operator
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")

while(1):
    #输入
    suanfa=input("请输入要选择的算法(DBSCAN 或 OPTICS 或 BIRCH 或 KMeans 或 MeanShift 或 GMM):")
    shujuji=input("请输入要选择的数据集（1或2或3）:")
    #准备数据集
    if shujuji=="1":
        centers = [[1, 1], [-1, -1], [1, -1]]  # 生成聚类中心点
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0) # 生成样本数据集
        X = StandardScaler().fit_transform(X) # StandardScaler作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
    else:
        X = datasets.load_iris()
        labels_true=X.target
        X=X.data
    #设置参数
    if shujuji=="1":
        if suanfa=="BIRCH":
            B1=3
            B2=0.2
            B3=50
        elif suanfa=="OPTICS":
            O1=26
            O2=0.4
        elif suanfa=="DBSCAN":
            D1=0.3
            D2=10
        elif suanfa=="KMeans":
            K=3
        elif suanfa=="GMM":
            G=3
    if shujuji=="2":
        if suanfa=="BIRCH":
            B1=3
            B2=0.4
            B3=50
        elif suanfa=="OPTICS":
            O1=10
            O2=1
        elif suanfa=="DBSCAN":
            D1=0.9
            D2=5
        elif suanfa=="KMeans":
            K=3
        elif suanfa=="GMM":
            G=3

    #调用程序
    if suanfa=="DBSCAN":
        # 调用密度聚类  DBSCAN
        db = DBSCAN(eps=D1, min_samples=D2).fit(X)
        # print(db.labels_)  # db.labels_为所有样本的聚类索引，没有聚类索引为-1
        # print(db.core_sample_indices_) # 所有核心样本的索引
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # 设置一个样本个数长度的全false向量
        core_samples_mask[db.core_sample_indices_] = True #将核心样本部分设置为true
        labels = db.labels_
        # 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Plot result
        # 使用黑色标注离散点
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:  # 聚类结果为-1的样本为离散点
                # 使用黑色绘制离散点
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)  # 将所有属于该聚类的样本位置置为true

            xy = X[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]  # 将所有属于该类的非核心样本取出，使用小图标绘制
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

        plt.title=('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    elif suanfa=="OPTICS":
        def compute_squared_EDM(X):
            return squareform(pdist(X,metric='euclidean'))
        # 显示决策图
        def plotReachability(data,eps):
            plt.figure()
            plt.plot(range(0,len(data)), data)
            plt.plot([0, len(data)], [eps, eps])
            plt.show()
        # 显示分类的类别
        def plotFeature(data,labels):
            clusterNum = len(set(labels))
            fig = plt.figure()
            scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
            ax = fig.add_subplot(111)
            for i in range(-1, clusterNum):
                colorSytle = scatterColors[i % len(scatterColors)]
                subCluster = data[np.where(labels == i)]
                ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12)
            plt.show()
        def updateSeeds(seeds,core_PointId,neighbours,core_dists,reach_dists,disMat,isProcess):
            # 获得核心点core_PointId的核心距离
            core_dist=core_dists[core_PointId]
            # 遍历core_PointId 的每一个邻居点
            for neighbour in neighbours:
                # 如果neighbour没有被处理过，计算该核心距离
                if(isProcess[neighbour]==-1):
                    # 首先计算改点的针对core_PointId的可达距离
                    new_reach_dist = max(core_dist, disMat[core_PointId][neighbour])
                    if(np.isnan(reach_dists[neighbour])):
                        reach_dists[neighbour]=new_reach_dist
                        seeds[neighbour] = new_reach_dist
                    elif(new_reach_dist<reach_dists[neighbour]):
                        reach_dists[neighbour] = new_reach_dist
                        seeds[neighbour] = new_reach_dist
            return seeds
        def OPTICS(data,eps=np.inf,minPts=15):
            # 获得距离矩阵
            orders = []
            disMat = compute_squared_EDM(data)
            # 获得数据的行和列(一共有n条数据)
            n, m = data.shape
            # np.argsort(disMat)[:,minPts-1] 按照距离进行 行排序 找第minPts个元素的索引
            # disMat[np.arange(0,n),np.argsort(disMat)[:,minPts-1]] 计算minPts个元素的索引的距离
            temp_core_distances = disMat[np.arange(0,n),np.argsort(disMat)[:,minPts-1]]
            # 计算核心距离
            core_dists = np.where(temp_core_distances <= eps, temp_core_distances, -1)
            # 将每一个点的可达距离未定义
            reach_dists= np.full((n,), np.nan)
            # 将矩阵的中小于minPts的数赋予1，大于minPts的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
            core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]
            # 用于标识是否被处理，没有被处理，设置为-1
            isProcess = np.full((n,), -1)
            # 遍历所有的核心点
            for pointId in core_points_index:
                # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
                if (isProcess[pointId] == -1):
                    # 将点pointId标记为当前类别(即标识为已操作)
                    isProcess[pointId] = 1
                    orders.append(pointId)
                    # 寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
                    neighbours = np.where((disMat[:, pointId] <= eps) & (disMat[:, pointId] > 0) & (isProcess == -1))[0]
                    seeds = dict()
                    seeds=updateSeeds(seeds,pointId,neighbours,core_dists,reach_dists,disMat,isProcess)
                    while len(seeds)>0:
                        nextId = sorted(seeds.items(), key=operator.itemgetter(1))[0][0]
                        del seeds[nextId]
                        isProcess[nextId] = 1
                        orders.append(nextId)
                        # 寻找newPoint种子点eps邻域（包含自己）
                        # 这里没有加约束isProcess == -1，是因为如果加了，本是核心点的，可能就变成了非和核心点
                        queryResults = np.where(disMat[:, nextId] <= eps)[0]
                        if len(queryResults) >= minPts:
                            seeds=updateSeeds(seeds,nextId,queryResults,core_dists,reach_dists,disMat,isProcess)
                        # 簇集生长完毕，寻找到一个类别
            # 返回数据集中的可达列表，及其可达距离
            return orders,reach_dists
        def extract_dbscan(data,orders, reach_dists, eps):
            # 获得原始数据的行和列
            n,m=data.shape
            # reach_dists[orders] 将每个点的可达距离，按照有序列表排序（即输出顺序）
            # np.where(reach_dists[orders] <= eps)[0]，找到有序列表中小于eps的点的索引，即对应有序列表的索引
            reach_distIds=np.where(reach_dists[orders] <= eps)[0]
            # 正常来说：current的值的值应该比pre的值多一个索引。如果大于一个索引就说明不是一个类别
            pre=reach_distIds[0]-1
            clusterId=0
            labels=np.full((n,),-1)
            for current in reach_distIds:
                # 正常来说：current的值的值应该比pre的值多一个索引。如果大于一个索引就说明不是一个类别
                if(current-pre!=1):
                    # 类别+1
                    clusterId=clusterId+1
                labels[orders[current]]=clusterId
                pre=current
            return labels
        X = np.array(X)
        orders,reach_dists=OPTICS(X,np.inf,O1)
        plotReachability(reach_dists[orders],1)
        labels=extract_dbscan(X,orders,reach_dists,O2)
        plotFeature(X,labels)
    elif suanfa=="BIRCH":
        labels = Birch(n_clusters = B1, threshold=B2, branching_factor=B3).fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.title=("BIRTCH Clusters")
        plt.show()
        print("CH指标:", metrics.calinski_harabaz_score(X, labels))
    elif suanfa=="KMeans":
        kmeans = KMeans(n_clusters=K, max_iter=300, n_init=10, init='k-means++', random_state=0)
        labels = kmeans.fit_predict(X)
        y_kmeans=labels
        for i in range(1,len(labels)):
            if y_kmeans[i] == 0:
                plt.scatter(X[i, 0], X[i, 1], s=15, c='red')
            elif y_kmeans[i] == 1:
                plt.scatter(X[i, 0], X[i, 1], s=15, c='blue')
            elif y_kmeans[i] == 2:
                plt.scatter(X[i, 0], X[i, 1], s=15, c='green')
            elif y_kmeans[i] == 3:
                plt.scatter(X[i, 0], X[i, 1], s=15, c='cyan')
            elif y_kmeans[i] == 4:
                plt.scatter(X[i, 0], X[i, 1], s=15, c='magenta')
        print(kmeans.cluster_centers_)
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 30, c = 'yellow', label = 'Centroids')
        plt.title=("K-Means Clusters")
        plt.show()
    elif suanfa=="MeanShift":
        ms=MeanShift(bin_seeding=True)#带宽
        labels=ms.fit_predict(X)
        #labels=ms.labels_
        plt.scatter(X[:,0],X[:,1],c=labels,cmap='prism')
        plt.title=('mean-shift Clusters')
    elif suanfa=="GMM":
        gmm = GaussianMixture(n_components=1).fit(X)
        labels = gmm.predict(X)
        plt.title=('高斯混合聚类')
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
        from sklearn import metrics
        print("CH指标:", metrics.calinski_harabaz_score(X, labels))
    else:
        print("输入错误！请根据提示输入。")


    # 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 模型评估
    print('估计的聚类个数为: %d' % n_clusters_)
    print("同质性: %0.3f" % metrics.homogeneity_score(labels_true, labels))  # 每个群集只包含单个类的成员。
    print("完整性: %0.3f" % metrics.completeness_score(labels_true, labels))  # 给定类的所有成员都分配给同一个群集。
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))  # 同质性和完整性的调和平均
    print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels))
    print("CH指标:", metrics.calinski_harabaz_score(X, labels))
    a=input("请选择是否继续(y/n):")
    if a == 'y':
        continue
    else:
        break
print("感谢使用！")
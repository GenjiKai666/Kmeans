import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, times):
        self.k = k
        self.times = times
    def fit(self, X):
        X = np.asarray(X)
        np.random.seed(0)
        self.clurter_centers = X[np.random.randint(0, len(X), self.k)]
        self.labels_ = np.zeros(len(X))

        for t in range(self.times):
            for index, x in enumerate(X):
                dis = np.sqrt(np.sum((x - self.clurter_centers) ** 2, axis=1))
                self.labels_[index] = dis.argmin()
            for i in range(self.k):
                self.clurter_centers[i] = np.mean(X[self.labels_ == i], axis=0)

    def predict(self, X):
        X = np.asarray(X)
        result = np.zeros(len(X))
        for index, x in enumerate(X):
            dis = np.sqrt(np.sum((x-self.clurter_centers)**2, axis=1))
            result[index] = dis.argmin()
        return result
def main():
    iris = datasets.load_iris()
    data = iris.data
    km = KMeans(k=3, times=50)
    km.fit(data)
    result = km.predict([[4, 4, 0, 0], [0, 0, 4, 4], [6.7, 3.1, 5.2, 2.3]])
    data2 = data[:, :2]
    kmeans = KMeans(k=3, times=50)
    kmeans.fit(data2)
    plt.figure(figsize=(10, 10))
    plt.scatter(data2[kmeans.labels_ == 0][:, 0], data2[kmeans.labels_ == 0][:, 1], label='0')
    plt.scatter(data2[kmeans.labels_ == 1][:, 0], data2[kmeans.labels_ == 1][:, 1], label='1')
    plt.scatter(data2[kmeans.labels_ == 2][:, 0], data2[kmeans.labels_ == 2][:, 1], label='2')
    plt.scatter(kmeans.clurter_centers[:, 0], kmeans.clurter_centers[:, 1], marker="+", s=300)
    plt.title("聚类分析结果")
    plt.legend()  # 生成图例
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()

if __name__ == "__main__":
    main()
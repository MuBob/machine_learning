# 第四个代码：矩阵数据集，测试sklearn
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
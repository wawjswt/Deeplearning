# 为了能用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始， 而不是从那些准备好的张量格式数据开始。
# 在Python中常用的数据分析工具中，我们通常使用pandas软件包。
import pandas as pd
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')  # data_file='../data/house_tiny.csv'
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

if __name__ == '__main__':
    data = pd.read_csv(data_file)
    print(data)
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, -1]
    print(inputs, '\n\n\n', outputs)

# 可以看到，在数据中有NaN缺失数值，处理缺失值的方法包括插值和删除等，其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。

    inputs.iloc[:,0] = inputs.fillna(inputs.iloc[:,0].mean())  # fillna函数用于补充缺失值, 把数据行的缺省项换成数据行的平均值
    print(inputs, '\n\n\n', outputs)
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)
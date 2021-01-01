import random
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("public_dataset/train.csv")
data.loc[data['sex'] == 'male', ['sex']] = 1
data.loc[data['sex'] == 'female', ['sex']] = 0
data.loc[data['smoker'] == 'yes', ['smoker']] = 1
data.loc[data['smoker'] == 'no', ['smoker']] = 0
data.loc[data['region'] == 'southeast', ['region']] = 1
data.loc[data['region'] == 'southwest', ['region']] = 2
data.loc[data['region'] == 'northeast', ['region']] = 3
data.loc[data['region'] == 'northwest', ['region']] = 4

x1 = data['age']
x2 = data['sex']
x3 = data['bmi']
x4 = data['children']
x5 = data['smoker']
x6 = data['region']
y = data['charges']

# 数据分析
res = []
t1 = data[data['smoker'] == 1]
t2 = data[data['smoker'] == 0]
res.append(t1['charges'].mean())
res.append(t2['charges'].mean())
plt.bar([1, 0], res, width=0.2)
plt.title('吸烟对医疗花费的影响')
# plt.show()

res = []
t3 = data[data['sex'] == 1]
t4 = data[data['sex'] == 0]
res.append(t3['charges'].mean())
res.append(t4['charges'].mean())
plt.bar([1, 0], res, width=0.2)
plt.title('性别对医疗花费的影响')
# plt.show()

res = []
t1 = data[data['children'] == 0]
t2 = data[data['children'] == 1]
t3 = data[data['children'] == 2]
t4 = data[data['children'] == 3]
t5 = data[data['children'] == 4]
t6 = data[data['children'] == 5]
res.append(t1['charges'].mean())
res.append(t2['charges'].mean())
res.append(t3['charges'].mean())
res.append(t4['charges'].mean())
res.append(t5['charges'].mean())
res.append(t6['charges'].mean())
plt.bar([0, 1, 2, 3, 4, 5], res, width=0.3)
plt.title('子女个数对医疗花费的影响')
# plt.show()

res = []
t1 = data[data['region'] == 1]
t2 = data[data['region'] == 2]
t3 = data[data['region'] == 3]
t4 = data[data['region'] == 4]
res.append(t1['charges'].mean())
res.append(t2['charges'].mean())
res.append(t3['charges'].mean())
res.append(t4['charges'].mean())
plt.bar([1, 2, 3, 4], res, width=0.3)
plt.title('地区对医疗花费影响')
#plt.show()

t1 = data['age'].values
t2 = data['charges'].values
plt.scatter(t1, t2)
plt.title('年龄对医疗花费影响')
#plt.show()

# print(data)

# 计算模型
data.drop(['charges'], axis=1, inplace=True)
data.drop(['region'], axis=1, inplace=True)  # 去除区域列
data.drop(['sex'], axis=1, inplace=True)  # 去除性别列

x = data.values
y = y.values

w = []
for i in range(10):
    xi = []
    yi = []
    for j in range(500):
        k = random.randint(0, 1069)
        xi.append(x[k])
        yi.append(y[k])
    xt = np.transpose(xi)
    A = np.dot(xt, xi)  # XT.*X
    A = A.astype(np.float)
    A = np.linalg.inv(A)  # (XT.*X)^-1
    w.append(np.dot(A, np.dot(xt, yi)))
print('w = ', end='')
# print(w)
W = []
for j in range(4):
    s = 0
    for i in range(10):
        s += w[i][j]
    W.append(s/10)
print(W)
# 预测
test = pd.read_csv("public_dataset/test_sample.csv")
test.drop(['charges'], axis=1, inplace=True)
test1 = test.copy()
test.loc[test['sex'] == 'male', ['sex']] = 1
test.loc[test['sex'] == 'female', ['sex']] = 0
test.loc[test['smoker'] == 'yes', ['smoker']] = 1
test.loc[test['smoker'] == 'no', ['smoker']] = 0
test.loc[test['region'] == 'southeast', ['region']] = 1
test.loc[test['region'] == 'southwest', ['region']] = 2
test.loc[test['region'] == 'northeast', ['region']] = 3
test.loc[test['region'] == 'northwest', ['region']] = 4
test.drop(['region'], axis=1, inplace=True)
test.drop(['sex'], axis=1, inplace=True)
x = test.values
print(test)
res = []
for i in range(268):
    res.append(np.dot(W, x[i]))

xgm = xgb.XGBRegressor()
train_x = data.astype(float)
train_y = y.astype(float)
xgm.fit(train_x, train_y)
res = xgm.predict(test.astype(float))
test1.insert(6, 'charges', res)
test1.to_csv('submission.csv', index=False)
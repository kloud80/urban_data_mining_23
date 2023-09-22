import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""===============================================================================
국토교통부 실거래가 불러오기
http://rtdown.molit.go.kr/
아파트 2023년 8월 1달간 전국 실거래 정보
==============================================================================="""
data = pd.read_csv('03주차 데이터 분석 기초/data/아파트(매매)__실거래가_20230916101223.csv', sep=',', skiprows=15, encoding='cp949', dtype='str')

data.dtypes
data.columns
data.shape


data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '')
data['거래금액(만원)'] = data['거래금액(만원)'].astype('int')

data = data.rename(columns={'거래금액(만원)' : '거래가'})

data['거래가'].describe()
data['거래가'].hist(bins=100)
plt.show()

tmp = data['시군구'].value_counts()
tmp = tmp.reset_index()

data = data[data['거래가'] < 50000]



"""===============================================================================
데이터 모델을 통한 해석
==============================================================================="""

data['거래가'].plot()
plt.show()

total_error = {}

#평균
model = data['거래가'].mean()
error = data['거래가'] - model
error = error * error
total_error['mean'] = error.sum()

#중앙
model = data['거래가'].median()
error = data['거래가'] - model
error = error * error
total_error['median'] = error.sum()

#최빈
model = data.mode(numeric_only=True, axis=0).values[0,0]
error = data['거래가'] - model
error = error * error
total_error['mode'] = error.sum()


plt.scatter(total_error['mean'], [0])
plt.scatter(total_error['median'], [1])
# plt.scatter(total_error['mode'], [2])
plt.show()


model = data['거래가'].mean()
model = np.full([data.shape[0],1], model)

plt.plot(data['거래가'].values[:50])
plt.plot(model[:50, 0])
plt.show()

"""===============================================================================
데이터 추가
==============================================================================="""
data['전용면적(㎡)']
data = data.rename(columns={'전용면적(㎡)':'면적'})
data['면적'] = data['면적'].astype('float')

data.dtypes


from sklearn.linear_model import LinearRegression


lm = LinearRegression()

X = np.array(data['면적'].values)
X = np.random.random([26286,1])
Y = np.array(data['거래가'].values)

X = X[:, np.newaxis]
Y = Y[:, np.newaxis]


lm.fit(X, Y)

lm.intercept_
lm.singular_

Y_hat = lm.predict(X)

plt.plot(Y[:50])
plt.plot(Y_hat[:50])
plt.show()



"""===============================================================================
잔차 정규성
==============================================================================="""

plt.hist(Y, bins=100)
plt.show()


residuals = Y - Y_hat
plt.hist(residuals, bins=100)
plt.show()


"""===============================================================================
모델 설명력
==============================================================================="""
plt.title('TSS')
plt.plot(Y[:50])
plt.plot(model[:50])
plt.fill_between(range(50), Y[:50,0], model[:50,0], color='red', alpha=0.4)
plt.show()


plt.title('SSR')
plt.plot(Y[:50])
plt.plot(Y_hat[:50])
plt.fill_between(range(50), Y[:50,0], Y_hat[:50, 0], color='blue', alpha=0.4)
plt.show()


plt.title('SST vs SSR')
plt.plot(Y[:50])
plt.fill_between(range(50), Y[:50,0], model[:50,0], color='red', alpha=0.4)
plt.fill_between(range(50), Y[:50,0], Y_hat[:50, 0], color='blue', alpha=0.4)
plt.show()


SST = np.sum(np.power(Y - np.mean(Y), 2))
SSR = np.sum(np.power(Y - Y_hat, 2))

1 - SSR/SST


from sklearn.metrics import r2_score
r2_score(Y, Y_hat)


"""===============================================================================
모델 설명력
==============================================================================="""
data['건축년도'] = data['건축년도'].astype('float')
data['건축년도'] = data['건축년도'].fillna(1950)

lm = LinearRegression()

X = np.array(data[['면적','건축년도']].values)
Y = np.array(data['거래가'].values)

X = X[:]
Y = Y[:, np.newaxis]


lm.fit(X, Y)

lm.intercept_
lm.singular_

Y_hat = lm.predict(X)

plt.plot(Y[:50])
plt.plot(Y_hat[:50])
plt.show()


r2_score(Y, Y_hat)



"""===============================================================================
모델 설명력
==============================================================================="""
data['층'] = data['층'].astype('int')
data['시도'] = data['시군구'].apply(lambda x : x.split(' ')[0])

data_dummies = pd.get_dummies(data['시도'] )
data_dummies.columns
data = pd.concat([data, data_dummies], axis=1)

data['년식'] = 2023 - data['건축년도']
data['년식제곱'] = data['년식']  * data['년식']


lm = LinearRegression()

X = np.array(data[['면적', '년식', '년식제곱', '강원특별자치도', '경기도', '경상남도', '경상북도', '광주광역시', '대구광역시', '대전광역시', '부산광역시',
       '서울특별시', '세종특별자치시', '울산광역시', '인천광역시', '전라남도', '전라북도', '제주특별자치도', '충청남도',
       '충청북도']].values)
Y = np.array(data['거래가'].values)

X = X[:]
Y = Y[:, np.newaxis]


lm.fit(X, Y)

lm.intercept_
lm.singular_

Y_hat = lm.predict(X)

plt.title('r2 : ' + str(r2_score(Y, Y_hat)))
plt.plot(Y[:50])
plt.plot(Y_hat[:50])
plt.show()





"""===============================================================================
변수 설명력
==============================================================================="""
import statsmodels.api as sm
results = sm.OLS(Y, sm.add_constant(X)).fit()
print(results.summary())


"""===============================================================================
변수 설명력
==============================================================================="""

data['년식2'] = data['년식'] * (100 + 10 * np.random.random([data.shape[0]])) / 100
data['년식3'] = data['년식'] + 3 * np.random.random([data.shape[0]])
data['년식4'] = data['년식'] + 4 * np.random.random([data.shape[0]])

lm = LinearRegression()

X = np.array(data[['면적', '년식', '년식2', '년식3', '년식4', '년식제곱', '강원특별자치도', '경기도', '경상남도', '경상북도', '광주광역시', '대구광역시', '대전광역시', '부산광역시',
       '서울특별시', '세종특별자치시', '울산광역시', '인천광역시', '전라남도', '전라북도', '제주특별자치도', '충청남도',
       '충청북도']].values)
Y = np.array(data['거래가'].values)

X = X[:]
Y = Y[:, np.newaxis]


lm.fit(X, Y)

lm.intercept_
lm.singular_

Y_hat = lm.predict(X)

plt.title('r2 : ' + str(r2_score(Y, Y_hat)))
plt.plot(Y[:50])
plt.plot(Y_hat[:50])
plt.show()

results = sm.OLS(Y, sm.add_constant(X)).fit()
print(results.summary())
r2_score(Y, Y_hat)

0.6383473412012983
0.6383500553317636
0.6383724079620556
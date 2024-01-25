import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
import plotly.express as px
import numpy as np
set_log_level("ERROR")

#성남시를 기준으로 합니다. 읽어오는 데이터와 컬럼은 지역에 따라 변화합니다.

df = pd.read_csv("/content/성남시.csv")

df1=df.drop( ['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26',
       'Unnamed: 27', 'Unnamed: 28'],axis=1)

df1['ds'] = pd.to_datetime(df1['날짜'].astype(str), format='%Y.%m')

df2=df1.drop( ['날짜'],axis=1)

df2 = df2.rename(columns={"평당가":"y","사설학원수(개)":"사설학원수"})

#X변수 정의
x_col_lst = ['사설학원수', '매수우위지수', '입주물량', '가계대출금리', 'UIG', '출퇴근량']

#Y변수 정의
y_col_lst = ['y']

X=df2[x_col_lst]
Y=df2[y_col_lst]

X_diff=X.diff()
X_diff.columns=X.columns+'_diff'

X_diff2=X.diff(periods=2)
X_diff2.columns=X.columns+'_diff2'

#Y변수들에 대한 lag 데이터 생성 (2 lag 까지만)
Y_diff=Y.diff()
Y_diff.columns=Y.columns+'_diff'

Y_diff2=Y.diff(periods=2)
Y_diff2.columns=Y.columns+'_diff2'

#X데이터 결합
X=pd.concat([X,X_diff,X_diff2],axis=1)

#Y데이터 결합
Y=pd.concat([Y,Y_diff,Y_diff2],axis=1)

#lag 데이터를 사용하므로 1행 ~ 2행 제거
#X=X.drop([0,1],axis=0)
#Y=Y.drop([0,1],axis=0)

#전체 데이터셋 생성
df2_pret=pd.concat([df2['ds'][2:],X,Y],axis=1)
df2_pret=df2_pret.reset_index(drop=True)

#train test split
cutoff = "2022-01-01" #데이터 분할 기준
train = df2_pret[df2_pret['ds']<cutoff]
test = df2_pret[df2_pret['ds']>=cutoff]

col_lst=df2_pret.columns
col_lst=col_lst.drop(['ds','y'])
col_lst=list(col_lst)

train= train.drop_duplicates('ds',keep='first')

#모델 구조 설정
m = NeuralProphet(

growth='off', # 추세 유형 설정(linear, discontinuous, off 중 선택 가능)

yearly_seasonality=False, #년간 계절성 설정

weekly_seasonality=False, #주간 계절성 설정

batch_size=64,#배치 사이즈 설정

epochs=100,#학습 횟수 설정

learning_rate=0.1, # 학습률 설정

n_lags= 2, #lag를 2까지 사용하였으므로, lag를 2로 설정

num_hidden_layers=4, #히든 레이어 수 설정

d_hidden=8,#은닉층에 대한 차원 수 설정


)

#독립 변인(변수) 추가 및 정규화
m = m.add_lagged_regressor(names=col_lst, normalize="minmax") 

#학습 수행
metrics = m.fit(train, validation_df=test, progress='plot')

#metric 확인
print("SmoothL1Loss: ", metrics.SmoothL1Loss.tail(1).item())
print("MAE(Train): ", metrics.MAE.tail(1).item())
print("MAE(Test): ", metrics.MAE_val.tail(1).item())
from keras.backend import dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import gradient_descent_v2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import SGD
from keras.metrics import MeanSquaredError
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

#해당 시 데이터 가져오기
Gung=pd.read_csv("/content/sample_data/화성시_LTV.csv")

#시드값 고정
tf.random.set_seed(14)
np.random.seed(seed=14)

#정규화 함수
def ts_train_test_normalize(all_data, time_steps, for_periods):
    #데이터 분할
    ts_train = all_data[:103].iloc[:,0:1].values
    ts_test = all_data[103:].iloc[:,0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    #정규화
    sc = MinMaxScaler(feature_range=(0,1))
    ts_train_scaled = sc.fit_transform(ts_train)

    #time_steps값에 따라 독립변수와 종속변수를 설정 EX)3,1이면 3개의 데이터의 묶음과 이후 1개의 데이터의 묶음을 각각 넣어줌
    X_train = []
    y_train = []
    for i in range(time_steps, ts_train_len-1):
        X_train.append(ts_train_scaled[i-time_steps:i, 0])
        y_train.append(ts_train_scaled[i:i+for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    #train 구조 재배열
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))

    #
    inputs = pd.concat((all_data["LTV"][:103], all_data["LTV"][103:]), axis=0).values
    inputs = inputs[len(inputs)-len(ts_test)-time_steps:]
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    #for_periods값에따라 이후 학습의 표본이 될 값의 수만큼씩 입력
    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i-time_steps:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, y_train, X_test, sc

#train, test 입력
X_train, y_train, X_test, sc = ts_train_test_normalize(Gung,3,1)
X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
y_train_see = pd.DataFrame(y_train)
pd.concat([X_train_see, y_train_see], axis = 1)
X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0], X_test.shape[1])))
pd.DataFrame(X_test_see)

#LSTM설정
my_LSTM_model = Sequential()
my_LSTM_model.add(LSTM(units = 50,
                      return_sequences = True,
                      input_shape = (X_train.shape[1],1),
                      activation = 'relu'))
my_LSTM_model.add(LSTM(units = 50, activation = 'relu'))
my_LSTM_model.add(Dense(units=1))

my_LSTM_model.compile(optimizer = SGD(lr = 0.01, decay = 1e-7,
                                    momentum = 0.9, nesterov = False),
                    loss = 'mean_squared_error')

my_LSTM_model.fit(X_train, y_train, epochs = 13000, batch_size =len(X_train), verbose = 0, shuffle=False)

#모델 적용
LSTM_prediction = my_LSTM_model.predict(X_test)
LSTM_prediction = sc.inverse_transform(LSTM_prediction)

#예측값 대입
y_pred = pd.DataFrame(LSTM_prediction[:, 0])
y_test=Gung.loc[103:,'LTV'][0:len(LSTM_prediction)]
y_test.reset_index(drop=True, inplace=True)

#'MAE','MSE','RMSE', 'RMSLE', 'R2'값을 알려주는 함수
def confirm_result(y_test, y_pred):
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MSLE = mean_squared_log_error(y_test, y_pred)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)

    pd.options.display.float_format = '{:.5f}'.format
    Result = pd.DataFrame(data=[MAE,MSE,RMSE, RMSLE, R2],
                        index = ['MAE','MSE','RMSE', 'RMSLE', 'R2'],
                        columns=['Results'])
    return Result
print(confirm_result(y_test, y_pred))
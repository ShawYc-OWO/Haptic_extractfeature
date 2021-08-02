from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import os

import time #helper libraries

# file is downloaded from finance.yahoo.com, 1.1.1997-1.1.2017
# training data = 1.1.1997 - 1.1.2007
# test data = 1.1.2007 - 1.1.2017
input_file=r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\Prepocessing\0.xls"
sum_file: List[Any] = []
df = []
data : List[Any] = []
data_x : List[Any] = []

def read_name():
    '''Read each file name from all Excel files for following work'''
    for root_dir, sub_dir, files in os.walk(r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\Prepocessing"):
        # count = 0
        for file in files:
            if file.endswith(".xls"):
                # Create absolute path
                # count += 1
                file_name = os.path.join(root_dir, file)
                sum_file.append(file_name)
    file_num = len(sum_file)
    return sum_file

# convert an array of values into a dataset matrix
def create_dataset(dataset_x,dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        #a = dataset_x[i:(i+look_back), :]
        a = dataset_x[i,:]
        dataX.append(a)
        # dataY.append(dataset[i + look_back, :])
        dataY.append(dataset[i,0])
    return np.array(dataX), np.array(dataY).reshape(-1,1)

if __name__ =='__main__':
    # fix random seed for reproducibility
    np.random.seed(5)
    file_name = read_name()
    # load the dataset
    for i in range(len(file_name)): # from 0 ~ len(sum_file)-1
        df.append([])

    for idx, name in enumerate(file_name):
        df[idx] = pd.read_excel(name, header=None,usecols=[1,2,3,4,5],names=None)
        tmp_data_y = df[idx][5].values
        tmp_data_x = (df[idx].values)[1:,0:4]
        data.append(np.array(tmp_data_y[1:]))
        data_x.append(np.array(tmp_data_x[:]))

    # take close price column[5]
    training_y = [i for item in data for i in item]
    training_x = [j for item in data_x for j in item]
    all_y = np.array(training_y)
    all_x = np.array(training_x)
    #all_y = df[5].values
    dataset = all_y.reshape(-1, 1)
    dataset_x = all_x.reshape(-1,4)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets, 80% test data, 20% training data
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_y, test_y = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train_x,test_x = dataset_x[0:train_size,:], dataset_x[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1, timestep 10
    look_back = 10 #尝试

    trainX, trainY = create_dataset(train_x,train_y, look_back)
    testX, testY = create_dataset(test_x,test_y, look_back)
    print(trainY.shape)


    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print(trainX.shape)


    # create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
    model = Sequential()
    model.add(LSTM(25, input_shape=(1,4))) # batch size为5 分别再加上是time, x,y coordinates and depth
    model.add(Dropout(0.1))
    model.add(Dense(1)) # Dense = 1
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=1000, batch_size=10, verbose=1) # 需要修改batchsize,尝试为10
    print('pass')

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY.flatten()])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY.flatten()])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    # print('testPrices:')
    print('trainForce:')
    testPrices=scaler.inverse_transform(dataset[test_size+look_back:])

    print('testPredictions:')
    print(testPredict)
    plt.plot(testPredictPlot)
    plt.show()
    exit()
    # export prediction and actual prices
    df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(testPrices.reshape(-1)), decimals=2)})
    df.to_csv("lstm_result.csv", sep=';', index=None)

    # plot the actual price, prediction in test data=red line, actual price=blue line

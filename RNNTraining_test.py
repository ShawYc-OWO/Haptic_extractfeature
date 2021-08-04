from typing import List, Any
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import os

import time #helper libraries



sum_file: List[Any] = []
df = []
df_test =[]
data : List[Any] = []
data_x : List[Any] = []
train_idx = []
test_idx = []
data_test =[]
data_test_x = []

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
    # return np.array(dataset_x), np.array(dataset).reshape(-1, 1)

if __name__ =='__main__':
    # 原先：fix random seed for reproducibility
    # 每次随机训练50%
    times = list(range(1))
    RMSE_record = []
    for i in times:
        RMSE_record.append([])
    for time in times:
        file_name = read_name()
        # print(file_name)
        # exit(0)
        # load the dataset
        for i in range(len(file_name)): # from 0 ~ len(sum_file)-1
            df.append([])

        '''For building training dataset'''
        for idx, name in enumerate(file_name):

            df[idx] = pd.read_excel(name, header=None, usecols=[1, 2, 3, 4, 5], names=None)
            tmp_data_y = df[idx].values[1:,4]
            tmp_data_x = (df[idx].values)[1:,0:4] # 含每次实验的time-step feature的
            # tmp_data_x = (df[idx].values)[1:,1:4]# 不含每次实验的time-step feature的
            # tmp_data_x = (df[idx].values)[1:, [0,3]] # 只含每次实验 timestep, 以及palpation depth
            data.append(np.array(tmp_data_y[:]))
            data_x.append(np.array(tmp_data_x[:]))

        tmp_y = []
        tmp_x = []

        '''For building test dataset'''
        # print(file_name[26])
        df_test = pd.read_excel(file_name[28], header=None, usecols=[1, 2, 3, 4, 5], names=None)
        tmp_y = (df_test.values)[1:,4]
        # tmp_x = (df_test.values)[1:,0:4]
        # tmp_data_x = (df_test.values)[1:, 1:4]  # 不含每次实验的time-step feature的
        tmp_data_x = (df_test.values)[1:,[0,3]] # 只含每次实验timestep, 以及palpation depth
        data_test.append(np.array(tmp_y[:]))
        data_test_x.append(np.array(tmp_x[:]))
        # print(tmp_data_y)
        # print(tmp_data_x)
        # plt.plot(tmp_data_x[:,3],tmp_data_y)


        # take close price column[5]
        training_y = [i for item in data for i in item]
        training_x = [j for item in data_x for j in item]
        testing_y = [i for i in data_test]
        testing_x = [j for j in data_test_x]
        all_y = np.array(training_y)
        all_x = np.array(training_x)
        all_test_y = np.array(testing_y)
        all_test_x = np.array(testing_x)
        #all_y = df[5].values
        dataset = all_y.reshape(-1, 1)
        dataset_test =all_test_y.reshape(-1,1)
        if all_x.shape[1] == 4:
            dataset_x = all_x.reshape(-1,4)
            dataset_test_x = all_test_x.reshape(-1,4)
        elif all_x.shape[1] == 3:
            dataset_x = all_x.reshape(-1, 3)
            dataset_test_x = all_test_x.reshape(-1,3)
        elif all_x.shape[1] == 2:
            dataset_x = all_x.reshape(-1, 2)
            dataset_test_x = all_test_x.reshape(-1, 2)

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        # dataset_test = scaler.fit_transform(dataset_test)
        # split into train and test sets, 50% test data, 5
        # 0% training data, for one palpation test, choose a single back and forth process
        train_size = int(len(dataset) * 0.50)
        fulllength = list(range(len(dataset)))
        train_idx = random.sample(fulllength,train_size)

        for i in fulllength:
            if i not in train_idx:
                test_idx.append(i)
        # print(len(test_idx))

        # test_size = len(dataset) - train_size
        train_y, test_y = dataset[train_idx, :], dataset[test_idx,:]
        train_x, test_x = dataset_x[train_idx, :], dataset_x[test_idx,:]

        # train_y, test_y = dataset[0:train_size,:], dataset[len(dataset)-306:len(dataset)-243,:]
        # train_x,test_x = dataset_x[0:train_size,:], dataset_x[len(dataset)-306:len(dataset)-243,:]
        '''train_y, test_y = dataset[0:train_size,:],dataset_test
        train_x, test_x = dataset_x[0:train_size, :],dataset_test_x'''

        # reshape into X=t and Y=t+1, timestep 10
        look_back = 10 #尝试

        trainX, trainY = create_dataset(train_x,train_y, look_back)
        testX, testY = create_dataset(test_x,test_y, look_back)
        # testX,testY = test_x, test_y
        # print(trainY.shape)


        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


        '''create and fit the LSTM network, optimizer=adam'''
        train_epoch = 1500
        model = Sequential()
        if all_x.shape[1] == 4:
            model.add(LSTM(100, input_shape=(1,4))) # 以及输入维度是time, x,y coordinates and depth,四维
            model.add(Dropout(0.05))
        elif all_x.shape[1] == 3:
            model.add(LSTM(88, input_shape=(1,3))) # 以及输入维度是 x,y coordinates and depth，三维
            model.add(Dropout(0.01)) # 这里修改为0.03
        elif all_x.shape[1] == 2:
            model.add(LSTM(49, input_shape=(1, 2)))  # 以及输入维度是 x,y coordinates and depth，三维
            model.add(Dropout(0.03))  # 这里修改为0.03
        model.add(Dense(1)) # Dense = 1
        model.compile(loss='mse', optimizer='adam')
        if time == 1:
            batch = 150
        else:
            batch = 300
        if all_x.shape[1] == 4:
            model.fit(trainX, trainY, epochs=train_epoch, batch_size=150, verbose=1) # 需要修改batchsize
        else:
            model.fit(trainX,trainY,epochs=train_epoch, batch_size=150, verbose=1) # 需要修改batchsize


        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)


        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY.flatten()])
        if len(testY) >= 80:
            testPredict = scaler.inverse_transform(testPredict)
        else:
            testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY.flatten()])
        # print(trainPredict)
        # print(testPredict)
        # exit()

        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))

        RMSE_record[time].append(['TrainRMSE:',trainScore,'TestRMSE:',testScore])

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        # testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        # plot baseline and predictions
        plt.figure(1)
        base_curve, = plt.plot(scaler.inverse_transform(dataset[train_idx, :]))
        # base_curve, = plt.plot(dataset_x[0:train_size,3],scaler.inverse_transform(dataset[0:train_size,:]))
        # train_predict_curve, = plt.plot(trainPredictPlot)
        # train_predict_curve, = plt.plot(dataset_x[0:train_size,3],trainPredict)
        train_predict_curve, = plt.plot(trainPredict)
        # plt.xlabel('Time step')
        plt.xlabel('Time step')
        plt.ylabel('Force')
        plt.legend([base_curve,train_predict_curve],['Basecurve','Prediction'])
        # print('testPrices:')
        print('trainForce:')
        # testPrice = scaler.inverse_transform(dataset[test_size+look_back:])

        plt.figure(2)
        time_step = np.array(list(range(len(testX)))).reshape(-1,1)
        # ax = plt.subplot(111, projection='3d')
        if all_x.shape[1] == 4:
            a = np.array([i for i in testX[:, :, 3].reshape(-1,1)])
        elif all_x.shape[1] == 3:
            a = np.array([i for i in testX[:,:,2].reshape(-1,1)])
        b = np.array([i for i in testY.reshape(-1, 1)])
        c = np.array([i for i in time_step])
        d = np.array([i for i in testPredict.reshape(-1, 1)])
        # print('testbasePredictions:')
        # print(b)
        # print('testY:')
        # print(d)
        # print('Palpation depth')
        # print(a)
        if all_x.shape[1] == 4:
            # base_curve, = plt.plot(dataset_test_x[:,3],scaler.inverse_transform(dataset_test[:]))
            base_curve, =plt.plot(c,b)
            # base_curve = ax.scatter3D(c,b,a)
            test_predict_curve, = plt.plot(c,d)
                # test_predict_curve = ax.scatter3D(c,d,a)
        # elif all_x.shape[1] == 3:
        #     base_curve, = plt.plot(dataset_test_x[:, 2], scaler.inverse_transform(dataset_test[:]))
        #     test_predict_curve, = plt.plot(dataset_test_x[:, 2], testPredict)
        # elif all_x.shape[1] == 2:
        #     base_curve, = plt.plot(dataset_test_x[:, 1], scaler.inverse_transform(dataset_test[:]))
        #     test_predict_curve, = plt.plot(dataset_test_x[:, 1], testPredict)
        # ax.set_xlabel('Time step')
        # ax.set_zlabel('Palpation depth')
        # ax.set_ylabel('Force')
        # plt.xlabel('Palpation depth')
        plt.xlabel('Time step')
        plt.ylabel('Force')
        # ax.set_xlim(0,len(a)+10)
        plt.legend([base_curve,test_predict_curve],['Basecurve','Prediction'])
        plt.show()
        # exit()

        # # export prediction and actual prices
        # df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_predict": np.around(list(testPrice.reshape(-1)), decimals=2)})
        # df.to_csv("lstm_result.csv", sep=';', index=None)

        # plot the actual price, prediction in test data=red line, actual price=blue line

    print(RMSE_record)
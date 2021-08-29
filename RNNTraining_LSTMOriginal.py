from typing import List, Any
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import seaborn as sns
from pandas import read_csv
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,mean_absolute_percentage_error,max_error
from keras.layers.core import Dense, Activation, Dropout
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import time # helper libraries



sum_file: List[Any] = []
sum_test_file: List[Any] = []
df = []
df_test =[]
data : List[Any] = []
data_x : List[Any] = []
train_idx = [] # for storing the training points
test_idx = [] # for storing the test points, the training and testing points all totally dfferent
data_test =[]
data_test_x = []

def read_train_name():
    '''Read each file name from test Excel files for following work'''
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle"
            r"\Prepocessing\General movement for  approaches and inputs\Train"):
        for file in files:
            if file.endswith(".xls"):
                # Create absolute path
                file_name = os.path.join(root_dir, file)
                sum_file.append(file_name)
    file_num = len(sum_file)
    print(file_num)
    return sum_file

def read_test_name():
    '''Read each file name from test Excel files for following work'''
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle"
            r"\Prepocessing\General movement for  approaches and inputs\Test"):
        for file in files:
            if file.endswith(".xls"):
                # Create absolute path
                file_test_name = os.path.join(root_dir, file)
                sum_test_file.append(file_test_name)
    file_num = len(sum_test_file)
    print(file_num)
    return sum_test_file


def create_dataset(x, y, seq_len=10):
    features = []
    targets = []

    for i in range(0, len(x)-seq_len, 1):
        data = x[i:i + seq_len]  # features
        label = y[i + seq_len]  # label
        # save features and lables
        features.append(data)
        targets.append(label)

    return np.array(features), np.array(targets)

if __name__ =='__main__':

    times = list(range(1))
    RMSE_record = []
    for i in times:
        RMSE_record.append([])
    for time in times:
        file_train_name = read_train_name()
        file_test_name = read_test_name()

        # load the dataset
        for i in range(len(file_train_name)): # from 0 ~ len(sum_file)-1
            df.append([])
        for i in range(len(file_test_name)):
            df_test.append([])

        '''For building training dataset'''
        for idx, name in enumerate(file_train_name):
            # df[idx] = pd.read_excel(name, header=None, usecols=[6, 7, 8, 9], names=None)
            df[idx] = pd.read_excel(name, header=None, usecols=[2, 3, 4, 5, 6, 7, 8, 9], names=None)
            tmp_data_y = df[idx].values[1:,7]
            tmp_data_x = (df[idx].values)[1:,0:7] # 含每次实验的time-step feature的
            # tmp_data_x = (df[idx].values)[1:,1:4]# 不含每次实验的time-step feature的
            data.append(np.array(tmp_data_y[:]))
            data_x.append(np.array(tmp_data_x[:]))

        tmp_y = []
        tmp_x = []

        '''For building test dataset'''
        for idx, name in enumerate(file_test_name):
            # df_test[idx] = pd.read_excel(name, header=None, usecols=[6, 7, 8, 9], names=None)
            df_test[idx] = pd.read_excel(name, header=None, usecols=[2, 3, 4, 5, 6, 7, 8, 9], names=None)
            tmp_datatest_y = df_test[idx].values[1:,7]
            tmp_datatest_x = (df_test[idx].values)[1:,0:7] # 含每次实验的time-step feature的
            data_test.append(np.array(tmp_datatest_y[:]))
            data_test_x.append(np.array(tmp_datatest_x[:]))


        training_y = [i for item in data for i in item]
        training_x = [j for item in data_x for j in item]
        testing_y = [i for item in data_test for i in item]
        testing_x = [j for item in data_test_x for j in item]


        all_y = np.array(training_y)
        all_x = np.array(training_x)
        all_test_y = np.array(testing_y)
        all_test_x = np.array(testing_x)

        dataset = all_y.reshape(-1, 1)
        dataset_test =all_test_y.reshape(-1,1)

        if all_x.shape[1] == 7:
            dataset_x = all_x.reshape(-1,7)
            dataset_test_x = all_test_x.reshape(-1,7)
        elif all_x.shape[1] == 3:
            dataset_x = all_x.reshape(-1, 3)
            dataset_test_x = all_test_x.reshape(-1,3)

        # normalize the dataset label
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # dataset = scaler.fit_transform(dataset)
        # joblib.dump(scaler, 'scaler01')
        # scaler_test = joblib.load('scaler01')
        # dataset_test = scaler_test.fit_transform(dataset_test)


        # split into train and test sets, 针对不同情况分成不同训练和测试集
        test_size = len(dataset_test)
        # fulllength = list(range(len(dataset)))
        # train_idx = random.sample(fulllength,train_size) # for ensuring the training points non-repetitive

        '''This is for ensuring the test data is totally different from training'''
        # for i in fulllength:
        #     if i not in train_idx:
        #         test_idx.append(i)


        # test_size = len(dataset) - train_size
        # train_y, test_y = dataset[train_idx, :], dataset[test_idx,:]
        train_y, test_y = dataset[:], dataset_test[:]
        # train_x, test_x = dataset_x[train_idx, :], dataset_x[test_idx,:]
        train_x, test_x = dataset_x[:], dataset_test_x[:]

        # train_y, test_y = dataset[0:train_size,:],dataset_test
        # train_x, test_x = dataset_x[0:train_size, :],dataset_test_x


        look_back_seq = 15
        train_feature, train_label = create_dataset(train_x,train_y, look_back_seq)
        test_feature, test_label = create_dataset(test_x,test_y, look_back_seq)


        # testX,testY = test_x, test_y
        # print(trainY.shape)
        # reshape input to be [samples, time steps, features]
        # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


        '''create and fit the LSTM network, optimizer=adam'''
        train_epoch = 200
        batch = 150
        model = Sequential()
        if all_x.shape[1] == 7:
            model.add(LSTM(units=256, input_shape=(look_back_seq,7),return_sequences=False)) # 以及输入维度是 x,y coordinates and depth,四元数
            # model.add(Dropout(0.1))
            # model.add(LSTM(units=128,return_sequences=True))
            # model.add(Dropout(0.03))
            # model.add(LSTM(units=64, return_sequences=True))
            # model.add(LSTM(units=128))

        elif all_x.shape[1] == 3:
            model.add(LSTM(128, input_shape=(look_back_seq,3))) # 以及输入维度是 x,y coordinates and depth，三维
            # model.add(Dropout(0.05)) # 这里修改为0.03

        model.add(Dense(1)) # Dense = 1
        model.compile(loss='mse', optimizer='adam')

        '''Show the model frame'''
        utils.plot_model(model, 'model frame.png')

        '''Save the best checkpoint for model'''
        checkpoint_file = "best_model.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file, monitor='loss', mode='min', save_best_only=True,
                                              save_weights_only=True)

        history = model.fit(train_feature,train_label,epochs=train_epoch, batch_size=batch, verbose=1) # 需要修改batchsize


        # make predictions
        trainPredict = model.predict(train_feature)
        testPredict = model.predict(test_feature)

        # invert predictions
        # trainPredict = scaler.inverse_transform(trainPredict)
        # train_label = scaler.inverse_transform([train_label.flatten()])
        # print(train_label.shape)
        # print(trainPredict.shape)
        # testPredict = scaler_test.inverse_transform(testPredict)
        # test_label = scaler_test.inverse_transform([test_label.flatten()])

        plt.figure(figsize=(16, 8))
        plt.plot(history.history['loss'], label='train loss')
        # plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(loc='best')


        trainRMSE = math.sqrt(mean_squared_error(train_label[:], trainPredict[:,0]))
        print('Train RMSE error: %.2f RMSE' % (trainRMSE))
        testRMSE = math.sqrt(mean_squared_error(test_label[:], testPredict[:,0]))
        testMAE = mean_absolute_error(test_label[:], testPredict[:,0])
        testMAPE = mean_absolute_percentage_error(test_label[:], testPredict[:,0])
        testMAXE = max_error(test_label[:], testPredict[:,0])
        Rsquare_modelscore = r2_score(test_label[:],testPredict[:,0])
        print('Test RMSE: %.2f RMSE' % (testRMSE)) # show the model prediction root mean square error value
        print('Test MAE: %.2f MAE' % (testMAE)) # show the model prediction mean absolute average error
        print('Test MAPE: %.2f MAPE' % (testMAPE)) # show the model prediction mean absolute percentage error
        print('Test MAXE: %.2f MAXE' % (testMAXE)) # show the max error in model prediction
        print('Model Score:%.2f ' %(Rsquare_modelscore)) # show the model score in R^2 term

        RMSE_record[time].append(['TrainRMSE:',trainRMSE,'TestRMSE:',testRMSE])

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back_seq:len(trainPredict)+look_back_seq, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        # testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        # plot baseline and predictions
        plt.figure(2)
        # base_curve, = plt.plot(scaler.inverse_transform(dataset[train_idx, :]))
        base_curve, = plt.plot(train_label.reshape(-1,1))
        # base_curve, = plt.plot(dataset_x[0:train_size,3],scaler.inverse_transform(dataset[0:train_size,:]))
        # train_predict_curve, = plt.plot(trainPredictPlot)
        # train_predict_curve, = plt.plot(dataset_x[0:train_size,3],trainPredict)
        train_predict_curve, = plt.plot(trainPredict)
        # plt.xlabel('Time step')
        plt.xlabel('Sample Number')
        plt.ylabel('Force')
        plt.legend([base_curve,train_predict_curve],['Basecurve','Prediction'])
        # print('trainForce:')
        # testPrice = scaler.inverse_transform(dataset[test_size+look_back:])

        plt.figure(3)
        time_step = np.array(list(range(len(test_feature)))).reshape(-1,1)
        # ax = plt.subplot(111, projection='3d')
        if all_x.shape[1] == 7:
            a = np.array([i for i in test_feature[:, :, 3].reshape(-1,1)])
        elif all_x.shape[1] == 3:
            a = np.array([i for i in test_feature[:, :, 2].reshape(-1,1)])
        b = np.array([i for i in test_label.reshape(-1, 1)])
        b = b[28:228]
        c = np.array([i for i in time_step[28:228]])
        d = np.array([i for i in testPredict[28:228].reshape(-1, 1)])

        if all_x.shape[1] == 7 or 3:
            # base_curve, = plt.plot(dataset_test_x[:,3],scaler.inverse_transform(dataset_test[:]))
            base_curve, =plt.plot(c,b)
            # base_curve = ax.scatter3D(c,b,a)
            test_predict_curve, = plt.plot(c,d)
                # test_predict_curve = ax.scatter3D(c,d,a)

        plt.xlabel('Sample Number')
        plt.ylabel('Force')
        # ax.set_xlim(0,len(a)+10)
        plt.legend([base_curve,test_predict_curve],['Experimental Measurement','Prediction'])
        plt.show()




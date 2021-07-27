import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os



sum_file_pos = []
sum_file_pres = []
all_list_time = []
all_list_pos = []
all_list_pres = []
para_pres = 0.00002698211
all_list_diff_force = []
all_list_force = []
y_pres = []
z_force = []
gra = 9.81 # convert kg to N
y_pos = []

def cal_size(pressure_array):
    return len(pressure_array)

def read_time():
    '''Read all time sequence data from all Excel files'''
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\In excel format"):
        for file in files:
            if file.endswith(".xlsx"):
                # Create absolute path
                file_name = os.path.join(root_dir, file)
                sum_file_pos.append(file_name)
    file_num = len(sum_file_pos)
    print(file_num)
    '''Create a parent list for storing following child lists'''
    for i in range(file_num): # from 0 ~ len(sum_file)-1
        all_list_time.append([])

    '''Extract position data from each  Excel and store in each corresponding child list'''
    for index,file_dir in enumerate(sum_file_pos[0:(file_num)]):
        time_step = pd.read_excel(io=file_dir, usecols=[0], names=None)
        all_list_time[index] = np.array(time_step.values.tolist()).flatten()

    return all_list_time

def read_pos():
    '''Read all z-axis position data from all Excel files'''
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\In excel format"):
        for file in files:
            if file.endswith(".xlsx"):
                # Create absolute path
                file_name = os.path.join(root_dir, file)
                sum_file_pos.append(file_name)
    file_num = len(sum_file_pos)
    # print(file_num)

    '''Create a parent list for storing following child lists'''
    for i in range(file_num): # from 0 ~ len(sum_file)-1
        all_list_pos.append([])

    '''Extract position data from each  Excel and store in each corresponding child list'''
    for index,file_dir in enumerate(sum_file_pos[0:(file_num)]):
        pos = pd.read_excel(io=file_dir, usecols=[9, 10, 11], names=None)
        all_list_pos[index] = np.array(pos.values.tolist())

    # print(all_list_pos[1])
    return all_list_pos

def read_pres():
    '''Read each time step pressure data from all Excel files'''
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\In excel format"):
        for file in files:
            if file.endswith(".xlsx"):
                # Create absolute path
                file_name = os.path.join(root_dir, file)
                sum_file_pres.append(file_name)
    file_num = len(sum_file_pres)
    '''Create a parent list for storing following child lists'''
    for i in range(file_num):  # from 0 ~ len(sum_file)-1
        all_list_pres.append([])

    '''Extract position data from each  Excel and store in each corresponding child list'''
    for index, file_dir in enumerate(sum_file_pres[0:(file_num)]):
        pres = pd.read_excel(io=file_dir, usecols=[i for i in range(24,2313)], names=None)
        tmp_array = np.array(pres.values.tolist())
        all_list_pres[index] = np.array(np.sum(tmp_array,axis=1))
    # print((all_list_pres[1]))

    return all_list_pres, file_num

def comp_force(pres_array,file_num):
    '''Compute force from pressure difference'''
    for i in range(file_num):  # from 0 ~ len(sum_file)-1
        all_list_diff_force.append([])
        all_list_force.append([])

    for index in range(file_num):
        count = 0
        for j in pres_array[index]:
            np.array(all_list_diff_force[index].append(j*gra))# storing each time step force from pressure
            if count >= 1:
                all_list_force[index].append((all_list_diff_force[index][count]-all_list_diff_force[index][0])) # calculating force from pressure difference
            count += 1

    return all_list_force

def indexofMin(tmp_array):
    minindex = 0
    currentindex = 0
    while currentindex < len(tmp_array):
        if tmp_array[currentindex] < tmp_array[minindex]:
            minindex = currentindex
        currentindex += 1
    return minindex

def modif(file_num,pos,force):
    for i in range(file_num):
        y_pos.append([])
        z_force.append([])

    for i in range(file_num):
        count = 0
        # minindex = indexofMin(pos[i][:,1])
        # for j in pos[i][:(minindex+1),1]:
        for j in pos[i][:, 1]:
            if count >= 1:
                np.array(y_pos[i].append(abs(j - pos[i][0][1])))
            count += 1
        z_force[i] = force[i]
    # print(len(y_pos[0]))
    # print(len(z_force[0]))
    return y_pos, z_force

def plot_dataset(file_num,pos,force):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0,file_num, 1)

    # for i in range(file_num):
    #     y_pres.append([])
    #     z_force.append([])
    #
    # for i in range(file_num):
    #     count = 0
    #     # minindex = indexofMin(pos[i][:,1])
    #     # for j in pos[i][:(minindex+1),1]:
    #     for j in pos[i][:,1]:
    #         if count >= 1:
    #             np.array(y_pres[i].append(abs(j-pos[i][0][1])))
    #         count += 1
    #     z_force[i] = force[i]

    # ax = plt.subplot(111,projection='3d')

    # for index in range(5):
    #     ax.scatter(X[index], y_pres[index], z_force[index])  # 绘制数据点
    ax = plt.subplot(111)
    for index in range(5):
        ax.scatter(pos[index], force[index],cmap='rainbow')  # 绘制数据点
        ax.set_ylabel('force')
        ax.set_xlabel('depth')

    # ax.set_zlabel('force')  # 坐标轴
    # ax.set_ylabel('depth')
    # ax.set_xlabel('testcase_No')

    # plt.xlim((0,31))
    # my_x_ticks = np.arange(0, 31, 1)
    # plt.xticks(my_x_ticks)
    # ax.plot_surface(X, y_pres, z_force, rstride=1, cstride=1, cmap='rainbow')
    plt.show()



if __name__ =='__main__':
    file_all = []
    time_series = read_time()
    time_series = [[round(j,2) for j in time_series[i]] for i in range(len(time_series))]
    print('pass')
    pos = read_pos()
    pres_list,file_num = read_pres()
    com_force = comp_force(pres_list,file_num)
    x_co_pos,y_force = modif(file_num,pos,com_force) # Note: the position here is in vertical direction, for plotting, it is the x coordinates
    print('pass')

    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\Prepocessing"):
        for i in range(file_num):
            file_name = os.path.join(root_dir, str(i))
            file_name += str('.xls') # create file path by hand
            file_all.append(file_name)

    for i in range(file_num):
        zipped = zip(time_series[i],pos[i][:,0],pos[i][:,2],x_co_pos[i],y_force[i])
        name = file_all[i]
        data = pd.DataFrame(zipped)
        #writer = pd.ExcelWriter(r'D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\Prepocessing\num.xls')  # 写入Excel文件
        writer = pd.ExcelWriter(name)
        data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名

        writer.save()
        writer.close()

    # pos = pd.read_excel(
    #     io=r'D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project\Third Term\Test data\test_in_1cm_circle\In excel format\testtest_N0 14-06-2021 13-30.xlsx',
    #     usecols=[9, 10, 11], names=None)
    # pos_li1 = np.array(pos.values.tolist())
    # pres = pd.read_excel(
    #     io=r'D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project\Third Term\Test data\test_in_1cm_circle\In excel format\testtest_N0 14-06-2021 13-30.xlsx',
    #     usecols=[i for i in range(24, 2313)])
    # pres_li1 = np.array(pres.values.tolist())
    # sum_pres = np.sum(pres_li1,axis=1)  # compute each time step pressure (every row with 2288 data points on pressure map)

    # print(sum_pres)
    # print(cal_size(sum_pres))
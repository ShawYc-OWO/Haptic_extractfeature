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

def read_name():
    '''Read all time sequence data from all Excel files'''
    name_collection = []
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\path_Test"):
        for file in files:
            if file.endswith(".csv"):
                # Create absolute path
                file_name = os.path.join(root_dir, file)
                sum_file_pos.append(file_name)
                name_collection.append(file)
    file_num = len(sum_file_pos)
    print(file_num)
    '''Create a parent list for storing following child lists'''
    for i in range(file_num): # from 0 ~ len(sum_file)-1
        all_list_time.append([])

    '''Extract position data from each  Excel and store in each corresponding child list'''
    for index,file_dir in enumerate(sum_file_pos[0:(file_num)]):
        # time_step = pd.read_excel(io=file_dir, usecols=[0], names=None)
        time_step = pd.read_csv(file_dir, usecols=[0], names=None)
        all_list_time[index] = np.array(time_step.values.tolist()).flatten()

    return all_list_time, name_collection

def read_pos():
    '''Read all z-axis position data from all Excel files'''
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\path_Test"):
        for file in files:
            if file.endswith(".csv"):
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
        # pos = pd.read_excel(io=file_dir, usecols=[9, 10, 11], names=None)
        pos = pd.read_csv(file_dir, usecols=[1,2,3,4,5,6,7], names=None)
        all_list_pos[index] = np.array(pos.values.tolist())

    return all_list_pos

def read_force():
    '''Read each time step pressure data from all Excel files'''
    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\path_Test"):
        for file in files:
            if file.endswith(".csv"):
                # Create absolute path
                file_name = os.path.join(root_dir, file)
                sum_file_pres.append(file_name)
    file_num = len(sum_file_pres)
    '''Create a parent list for storing following child lists'''
    for i in range(file_num):  # from 0 ~ len(sum_file)-1
        all_list_force.append([])

    '''Extract position data from each  Excel and store in each corresponding child list'''
    for index, file_dir in enumerate(sum_file_pres[0:(file_num)]):
        # pres = pd.read_excel(io=file_dir, usecols=[i for i in range(24,2313)], names=None)
        force = pd.read_csv(file_dir, usecols=[8], names=None)
        tmp_array = np.array(force.values.tolist())
        all_list_force[index] = np.array(tmp_array).flatten()

    return all_list_force, file_num

# def comp_force(pres_array,file_num):
#     '''Compute force from pressure difference'''
#     for i in range(file_num):  # from 0 ~ len(sum_file)-1
#         all_list_diff_force.append([])
#         all_list_force.append([])
#
#     for index in range(file_num):
#         count = 0
#         for j in pres_array[index]:
#             np.array(all_list_diff_force[index].append(j*gra)) # storing each time step force from pressure
#             if count >= 1:
#                 all_list_force[index].append((all_list_diff_force[index][count]-all_list_diff_force[index][0])) # calculating force from pressure difference
#             count += 1
#
#     return all_list_force

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
        # count = 0
        # minindex = indexofMin(pos[i][:,1])
        # for j in pos[i][:(minindex+1),1]:
        y_pos[i] = pos[i]
        z_force[i] = force[i]
    print(len(y_pos[1]))
    print(len(z_force[1]))
    return y_pos, z_force

def plot_dataset(file_num,pos,force):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0,file_num, 1)

    # ax = plt.subplot(111,projection='3d')

    idx = list(range(file_num))
    for index in idx[:12]:
        ax = plt.subplot(3, 4, (index + 1))
        ax.scatter(pos[index], force[index])  # 绘制数据点
        ax.set_ylabel('force')
        ax.set_xlabel('depth')
    plt.show()
    for index in idx[36:47]:
        ax = plt.subplot(3, 4, (index -35))
        ax.scatter(pos[index], force[index])  # 绘制数据点
        ax.set_ylabel('force')
        ax.set_xlabel('depth')
    plt.show()
    for index in idx[47:59]:
        ax = plt.subplot(3,4, (index-46))
        ax.scatter(pos[index], force[index])  # 绘制数据点
        ax.set_ylabel('force')
        ax.set_xlabel('depth')
    plt.show()
    for index in idx[59:71]:
        ax = plt.subplot(3, 4, (index - 58))
        ax.scatter(pos[index], force[index])  # 绘制数据点
        ax.set_ylabel('force')
        ax.set_xlabel('depth')
    plt.show()
    for index in idx[71:82]:
        ax = plt.subplot(3, 4, (index - 70))
        ax.scatter(pos[index], force[index])  # 绘制数据点
        ax.set_ylabel('force')
        ax.set_xlabel('depth')
    plt.show()
    for index in idx[82:94]:
        ax = plt.subplot(3, 4, (index - 81))
        ax.scatter(pos[index], force[index])  # 绘制数据点
        ax.set_ylabel('force')
        ax.set_xlabel('depth')
    plt.show()


if __name__ =='__main__':
    file_all = []
    name_collection = []
    time_series, name_collection = read_name()
    time_series = [[round(j,2) for j in time_series[i]] for i in range(len(time_series))]
    print('pass')
    pos = read_pos()
    force_list,file_num = read_force()
    print('pass')

    # com_force = comp_force(force,file_num)
    # x_co_pos,y_force = modif(file_num,pos,force_list) # Note: the position here is in vertical direction, for plotting, it is the x coordinates
    # plot_dataset(file_num,x_co_pos,y_force)

    for root_dir, sub_dir, files in os.walk(
            r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\Prepocessing\Angle with path\Test"):
        for i in range(file_num):
            file_name = os.path.join(root_dir, str(name_collection[i]))
            file_name += str('.xls') # create new file path
            file_all.append(file_name)

    for i in range(file_num):
        zipped = zip(time_series[i],pos[i][:,3],pos[i][:,4],pos[i][:,5],pos[i][:,6],pos[i][:,0],pos[i][:,2],pos[i][:,1],force_list[i]) # storing information for training and testing
        name = file_all[i]
        data = pd.DataFrame(zipped)
        #writer = pd.ExcelWriter(r'D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\Prepocessing\num.xls')  # 写入Excel文件
        writer = pd.ExcelWriter(name)
        data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名

        writer.save()
        writer.close()

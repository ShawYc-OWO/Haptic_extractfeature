import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm,pinv
import ExtractPos_Pres as ext

np.random.seed(20)
class RBF:
    '''RBF neural network'''
    def __init__(self,input_dim, num_neuron, out_dim):
        self.input_dim = input_dim
        self.num_neuron = num_neuron
        self.out_dim = out_dim
        self.lamda = 8 # RBF的扩展常数，乘在欧式距离范数前的系数
        self.centres = [np.random.uniform(-1,1,input_dim) for i in range(num_neuron)]
        self.W = np.random.random((num_neuron,out_dim))

    def _basicFunc(self,c,x):
        return np.exp(-self.lamda * norm(c-x)**2)

    def _calAct(self,X):
        print(X.shape[0])
        G = np.zeros((X.shape[0],self.num_neuron),dtype='float')
        for c_idx ,c in enumerate(self.centres):
            for x_idx, x in enumerate(X):
                G[x_idx,c_idx] = self._basicFunc(c,x)
        return G

    def train(self,X,Y):
        x = np.array(X).reshape(len(X), 1)
        y = np.array(Y).reshape(len(Y), 1)
        # rdn_idx = np.random.permutation(X.shape[0])[:self.num_neuron]
        rdn_idx = np.random.permutation(x.shape[0])[:self.num_neuron]
        self.centres = [x[i,:] for i in rdn_idx]
        '''计算激活函数值'''
        G = self._calAct(x)
        self.W = np.dot(pinv(G),y)


    def predict(self,X):
        G = self._calAct(X)
        Y = np.dot(G,self.W)
        return Y

if __name__ == '__main__':
    # n = 100 # 小构造一个等差数列测试一下
    # x = np.linspace(-1,1,n).reshape(n,1)
    # y = np.sin(3*(x+0.5)**3-1)
    x_pos = ext.read_pos()
    y_pres_list, file_num = ext.read_pres()
    com_force = ext.comp_force(y_pres_list, file_num)
    x_pos,y_force = ext.modif(file_num,x_pos,com_force)

    # for i in range(np.array(x_pos,dtype='object').shape[0][:15]):
    rbf = RBF(1,60,1)
    for i in range(24):
        print('flag')
        # for j, x_val in enumerate(x_pos[i]):
        #     print(x_val)
        #     print(y_force[i][j])
        rbf.train(x_pos[i],y_force[i])
    print('pass')
    ori_test = np.array(x_pos[26]).reshape(len(x_pos[26]),1)
    test = rbf.predict(ori_test)

    # rbf = RBF(1,12,1)
    # rbf.train(x,y)
    # test = rbf.predict(x)
    #
    plt.plot(x_pos[26],y_force[26],'k-',label=u'actual value')
    plt.plot(x_pos[26],test,'r-',label=u'predict')
    plt.ylim(-1,5)
    plt.legend(loc='upper left')
    plt.show()


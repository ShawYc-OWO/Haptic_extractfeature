import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm,pinv

np.random.seed(20)
class RBF:
    '''RBF neural network'''
    def __init__(self,input_dim, num_neuron, out_dim):
        self.input_dim = input_dim
        self.num_neuron = num_neuron
        self.out_dim = out_dim
        self.lamda = 5 # RBF的扩展常数，乘在欧式距离范数前的系数
        self.centres = [np.random.uniform(-1,1,input_dim) for i in range(num_neuron)]
        self.W = np.random.random((num_neuron,out_dim))

    def _basicFunc(self,c,x):
        return np.exp(-self.lamda * norm(c-x)**2)

    def _calAct(self,X):
        G = np.zeros((X.shape[0],self.num_neuron),dtype='float')
        for c_idx ,c in enumerate(self.centres):
            for x_idx, x in enumerate(X):
                G[x_idx,c_idx] = self._basicFunc(c,x)
        return G

    def train(self,X,Y):
        # rdn_idx = np.random.permutation(X.shape[0])[:self.num_neuron]
        rdn_idx = np.random.permutation(X.shape[0])[:int(X.shape[0]/2)]
        self.centers = [X[i,:] for i in rdn_idx]
        '''计算激活函数值'''
        G = self._calAct(X)
        self.W = np.dot(pinv(G),Y)


    def predict(self,X):
        G = self._calAct(X)
        Y = np.dot(G,self.W)
        return Y

if __name__ == '__main__':
    n = 100 # 小构造一个等差数列测试一下
    x = np.linspace(-1,1,n).reshape(n,1)
    y = np.sin(3*(x+0.5)**3-1)

    rbf = RBF(1,12,1)
    rbf.train(x,y)
    test = rbf.predict(x)

    plt.plot(x,y,'k-',label=u'actual value')
    plt.plot(x,test,'r-',label=u'predict')
    plt.xlim(-1.2,1.2)
    plt.legend(loc='upper left')
    plt.show()


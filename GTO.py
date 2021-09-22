# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:37:54 2021

@author: zongsing.huang

https://doi.org/10.1002/int.22535
https://www.mathworks.com/matlabcentral/fileexchange/95953-artificial-gorilla-troops-optimizer
"""

import numpy as np
import matplotlib.pyplot as plt

class GTO():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0, 
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub*np.ones([self.P, self.D])
        self.lb = lb*np.ones([self.P, self.D])
        self.p = 0.03
        self.beta = 3
        self.w = 0.8
        
        self.pbest_X = np.zeros([self.P, self.D])
        self.pbest_F = np.zeros([self.P]) + np.inf
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
        
    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        # 適應值計算
        F = self.fitness(self.X)
        # 更新最佳解
        mask = F < self.pbest_F
        self.pbest_X[mask] = self.X[mask].copy()
        self.pbest_F[mask] = F[mask].copy()
        
        if np.min(F) < self.gbest_F:
            idx = F.argmin()
            self.gbest_X = self.X[idx].copy()
            self.gbest_F = F.min()
        
        self.loss_curve[0] = self.gbest_F
        
        # 迭代
        for g in range(1, self.G):
            r1 = np.random.uniform()
            r2 = np.random.uniform()
            a = ( np.cos(2*r1) + 1 ) * (1-g/self.G)
            C = a * (2*r2-1)
            
            for i in range(self.P):
                r3 = np.random.uniform()
                if r3<self.p:
                    self.X[i] = np.random.uniform(low=self.lb[0], high=self.ub[0])
                elif r3>=0.5:
                    r4 = np.random.uniform()
                    rand_X = self.pbest_X[np.random.randint(low=0, high=self.P)]
                    Z = np.random.uniform(low=-a, high=a, size=[self.D])
                    H = Z*self.pbest_X[i]
                    self.X[i] = (r4-a)*rand_X + C*H
                else:
                    r5 = np.random.uniform()
                    rand_X1 = self.X[np.random.randint(low=0, high=self.P)]
                    rand_X2 = self.X[np.random.randint(low=0, high=self.P)]
                    self.X[i] = self.pbest_X[i] - C * (C*(self.pbest_X[i]-rand_X1) + r5*(self.pbest_X[i]-rand_X2))

            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)
            
            # 適應值計算
            F = self.fitness(self.X)
            # 更新最佳解
            mask = F < self.pbest_F
            self.pbest_X[mask] = self.X[mask].copy()
            self.pbest_F[mask] = F[mask].copy()
            
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
        
            for i in range(self.P):
                if a>=self.w:
                    g_var = 2**C
                    delta = ( np.abs(self.X.mean())**g_var ) ** (1/g_var)
                    self.X[i] = C * delta * ( self.pbest_X[i]-self.gbest_X ) + self.pbest_X[i]
                else:
                    r6 = np.random.uniform()
                    if r6>=0.5:
                        h = np.random.normal(size=[self.D])
                    else:
                        h = np.random.normal()
                    
                    r7 = np.random.uniform()
                    self.X[i] = self.gbest_X - ( self.gbest_X*(2*r7-1) - self.pbest_X[i]*(2*r7-1) ) * (self.beta*h)
            
            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)
            
            # 適應值計算
            F = self.fitness(self.X)
            # 更新最佳解
            mask = F < self.pbest_F
            self.pbest_X[mask] = self.X[mask].copy()
            self.pbest_F[mask] = F[mask].copy()
            
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
        
            self.loss_curve[g] = self.gbest_F
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
            
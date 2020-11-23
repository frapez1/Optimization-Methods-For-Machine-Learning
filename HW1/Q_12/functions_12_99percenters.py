import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
import time


def phi_single_vect(T, sigma):
  return np.exp(-(T/sigma)**2)

def phi(x,c, sigma):
  return np.exp(-(np.linalg.norm(x-c)/sigma)**2)


def phi_primo(x, c, sigma):
  return 2*np.linalg.norm(x-c)*phi(x, c, sigma)/sigma**2

def Norm_matrix(X, centers, sigma):
  return np.array([[phi(X[:,k], centers[:,j], sigma) for k in range(X.shape[1])] for j in range(centers.shape[1])])

def RBF_supervised(omega, X, sigma, par, rho, Y):
  C = omega[:(par[0]*par[1])].reshape(par) 
  v = omega[(par[0]*par[1]):].reshape((par[1], 1))
  hidden = Norm_matrix(X, C, sigma)
  y_pred = np.dot(v.transpose(), hidden)
  e = (1/(2*Y.shape[1]))*np.linalg.norm(y_pred-Y)**2 + rho*np.linalg.norm(omega)**2
  return e


def RBF_supervised_test(omega, X, sigma, par, rho, Y):
  C = omega[:(par[0]*par[1])].reshape(par) 
  v = omega[(par[0]*par[1]):].reshape((par[1], 1))
  hidden = Norm_matrix(X, C, sigma)
  y_pred = np.dot(v.transpose(), hidden)
  e = (1/(2*Y.shape[1]))*np.linalg.norm(y_pred-Y)**2 
  return e


def JAC_RBF_supervised(omega, X, sigma, par, rho, Y):
  C = omega[:(par[0]*par[1])].reshape(par) 
  v = omega[(par[0]*par[1]):].reshape((par[1], 1))
  dC = np.zeros_like(C)
  dv = np.zeros_like(v)
  hidden = Norm_matrix(X, C, sigma)
  Y_diff = np.dot(v.transpose(), hidden) - Y
  for j in range(par[1]):
    v_j = v[j]
    dv[j] =  np.mean(Y_diff*hidden[j,:]) + 2*rho*v[j]
    for i in range(par[0]):
      X_ic_j = X[i,:] - C[i,j]
      dC[i,j] = np.mean(2*Y_diff*v_j*X_ic_j*phi_single_vect(X_ic_j, sigma)/sigma**2) + 2*rho*C[i,j]
  return np.concatenate((dC, dv), axis = None) 



def Grid_RBF(X_tot, X_test, y_tot, y_test, N_, Sigma, Rho, K, Res):
  for N in N_:
    for sigma in Sigma:
      for rho in Rho:
        e_val = 0
        e = 0
        for i in range(K):
          X_train, X_val, y_train, y_val = train_test_split(X_tot, y_tot, test_size = 0.2)
          X_train = np.transpose(X_train)
          X_val = np.transpose(X_val)
          y_train = y_train.reshape(1,len(y_train))
          y_val = y_val.reshape(1,len(y_val))


          V = np.random.randn(N)
          centers = X_train[:,np.random.choice(X_train.shape[1], size = N, replace = False)]
          par_c = centers.shape
          omega_RBF = np.concatenate((centers, V), axis = None)
          
          
          omega_RBF = minimize(RBF_supervised, x0 = omega_RBF, args = (X_train, sigma, par_c, rho, y_train),jac=JAC_RBF_supervised, method = 'L-BFGS-B', tol = 1e-9, options = {'maxiter': 100, 'disp': False})
          # print(scipy.optimize.OptimizeResult(omega_MLP))
          e += RBF_supervised_test(omega_RBF.x, X_train, sigma, par_c, rho, y_train)
          e_val += RBF_supervised_test(omega_RBF.x, X_val, sigma, par_c, rho, y_val)
        Res.append([N,sigma, rho, e/K, e_val/K ])
  return Res


def Plot_3d(X_ps1, X_ps2, Y_ps):
  fig = plt.figure()
  ax = fig.gca(projection = '3d')
  jet = plt.get_cmap('jet')
  surf = ax.plot_surface(X_ps1, X_ps2, Y_ps, rstride = 1, cstride = 1, cmap = jet, linewidth = 0)
  ax.set_zlim3d(Y_ps.min(), Y_ps.max())
  plt.show()

def RBF_supervised_plot(omega, X, sigma, par, rho):
  C = omega[:(par[0]*par[1])].reshape(par) 
  v = omega[(par[0]*par[1]):].reshape((par[1], 1))
  hidden = Norm_matrix(X, C, sigma)
  y_pred = np.dot(v.transpose(), hidden)
  return y_pred

def Plot_RBF(omega, sigma, par, rho, punti):
  X_p = np.linspace(-2, 2, punti)
  X_ps1 = np.concatenate([X_p for i in range(punti)])
  X_ps2 = np.concatenate([[X_p[j] for i in range(punti)] for j in range(punti)])
  X_ps = np.array([X_ps1, X_ps2])
  Y_ps = RBF_supervised_plot(omega, X_ps, sigma, par, rho).transpose().reshape(punti**2)
  X_ps1, X_ps2, Y_ps = X_ps1.reshape(punti, punti), X_ps2.reshape(punti, punti), Y_ps.reshape(punti, punti)
  Plot_3d(X_ps1, X_ps2, Y_ps)




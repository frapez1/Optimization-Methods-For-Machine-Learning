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



def g(t, sigma):
  return (np.exp(2*sigma*t) - 1)/(np.exp(2*sigma*t) + 1)

def g_primo(t, sigma):
  return (4*sigma*np.exp(2*sigma*t))/(np.exp(2*sigma*t) + 1)**2


def MLP(omega, X, sigma, par, rho, Y):
  w = omega[:(par[0]*par[1])].reshape(par)
  v = omega[(par[0]*par[1]):].reshape((par[1],1))
  y_pred = v.transpose().dot(g(w.transpose().dot(X), sigma))
  e = (1/(2*Y.shape[1]))*np.linalg.norm(y_pred-Y)**2 + rho*np.linalg.norm(omega)**2
  return e

def MLP_test(omega, X, sigma, par, rho, Y):
  w = omega[:(par[0]*par[1])].reshape(par)
  v = omega[(par[0]*par[1]):].reshape((par[1],1))
  y_pred = v.transpose().dot(g(w.transpose().dot(X), sigma))
  e = (1/(2*Y.shape[1]))*np.linalg.norm(y_pred-Y)**2 
  return e




def JAC_MLP_W(omega, X, sigma, par, rho, Y):
  w = omega[:(par[0]*par[1])].reshape(par)
  v = omega[(par[0]*par[1]):].reshape((par[1],1))
  dw = np.zeros_like(w)
  dv = np.zeros_like(v)
  W_tX = np.dot(w.transpose(), X)
  Y_diff = np.dot(v.transpose(),g(W_tX, sigma)) - Y
  for j in range(par[1]):
    gp_W_jX = g_primo(W_tX[j,:], sigma).transpose()
    for i in range(par[0]):
      v_jX_i = v[j]*X[i:]
      dw[i,j] = np.mean(Y_diff*v_jX_i*gp_W_jX) + 2*rho*w[i,j]
  return np.concatenate((dw, dv), axis = None) 

def JAC_MLP_v(omega, X, sigma, par, rho, Y):
  w = omega[:(par[0]*par[1])].reshape(par)
  v = omega[(par[0]*par[1]):].reshape((par[1],1))
  dw = np.zeros_like(w)
  dv = np.zeros_like(v)
  W_tX = np.dot(w.transpose(), X)
  Y_diff = np.dot(v.transpose(),g(W_tX, sigma)) - Y
  for j in range(par[1]):
    g_W_jX = g(W_tX[j,:], sigma).transpose()
    dv[j] =  np.mean(Y_diff*g_W_jX) + 2*rho*v[j]
  return np.concatenate((dw, dv), axis = None) 







def Plot_3d(X_ps1, X_ps2, Y_ps):
  fig = plt.figure()
  ax = fig.gca(projection = '3d')
  jet = plt.get_cmap('jet')
  surf = ax.plot_surface(X_ps1, X_ps2, Y_ps, rstride = 1, cstride = 1, cmap = jet, linewidth = 0)
  ax.set_zlim3d(Y_ps.min(), Y_ps.max())
  plt.show()


def MLP_plot(omega, X, sigma, par, rho):
  w = omega[:(par[0]*par[1])].reshape(par)
  v = omega[(par[0]*par[1]):].reshape((par[1],1))
  y_pred = v.transpose().dot(g(w.transpose().dot(X), sigma))
  return y_pred

def Plot_MLP(omega, sigma, par, rho, punti):
  X_p = np.linspace(-2, 2, punti)
  X_ps1 = np.concatenate([X_p for i in range(punti)])
  X_ps2 = np.concatenate([[X_p[j] for i in range(punti)] for j in range(punti)])
  uno = np.ones_like(X_ps1)
  X_ps = np.array([uno, X_ps1, X_ps2])
  Y_ps = MLP_plot(omega, X_ps, sigma, par, rho).transpose().reshape(punti**2)
  X_ps1, X_ps2, Y_ps = X_ps1.reshape(punti, punti), X_ps2.reshape(punti, punti), Y_ps.reshape(punti, punti)
  Plot_3d(X_ps1, X_ps2, Y_ps)






















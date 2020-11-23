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
from functions_22_99percenters import *
df = pd.read_csv('OMML2020_Assignment_1_Dataset.csv')


np.random.seed(1908315)

Y = df['Y'].to_numpy()
X_RBF = df[['X_1','X_2']].to_numpy()



X_tot, X_test, y_tot, y_test = train_test_split(X_RBF, Y, test_size = 0.15)
# define train and test
X_tot_ = np.transpose(X_tot)
X_test_ = np.transpose(X_test)
y_test_ = y_test.reshape(1,len(y_test))
y_tot_ = y_tot.reshape(1,len(y_tot))


# size 
input_size = 2 
output_size = 1



#####################################################################
##################### cambiare valori in base a 11 #######################
#####################################################################


N = 40
sigma = 0.5
rho = 0.00001


# random centers and weigths
V = np.random.randn(N)
centers = centers(N, np.transpose(X_tot_))
par_c = centers.shape
omega_RBF = np.concatenate((centers, V), axis = None)

# optimize
start = time.time()
omega_RBF = minimize(RBF_supervised, x0 = omega_RBF, args = (X_tot_, sigma, par_c, rho, y_tot_),jac=JAC_RBF_supervised_v, method = 'L-BFGS-B',tol = 1e-9, options = {'maxiter': 100, 'disp': False})
delta_t = time.time() - start 

# evaluate error
e = RBF_supervised_test(omega_RBF.x, X_tot_, sigma, par_c, rho, y_tot_)
e_val = RBF_supervised_test(omega_RBF.x, X_test_, sigma, par_c, rho, y_test_)


what = ['N','sigma','rho', 'tollerance', 'max_numb_iter', 'optimz_solver', 'nfev', 'niter', 'time', 'train_error', 'test_error' ]
values_ = [N, sigma, rho, 1e-9, 100, 'L-BFGS-B', omega_RBF.nfev, omega_RBF.nit, delta_t,  e, e_val]



#####################################################################
##################### print in maniera decente #######################
#####################################################################

for i in range(len(what)):
    print('{} = {}'.format(what[i], values_[i]))




# plot figure
Plot_RBF(omega_RBF.x, sigma, (2,N), rho, 100)








